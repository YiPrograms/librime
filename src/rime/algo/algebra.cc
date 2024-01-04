//
// Copyright RIME Developers
// Distributed under the BSD License
//
// 2012-01-19 GONG Chen <chen.sst@gmail.com>
//
#include <tuple>
#include <algorithm>
#include <fstream>
#include <shared_mutex>
#include <rime/algo/algebra.h>
#include <rime/algo/calculus.h>

#include <nvtx3/nvtx3.hpp>
#include <omp.h>

namespace rime {

bool Script::AddSyllable(const string& syllable) {
  if (find(syllable) != end())
    return false;
  Spelling spelling(syllable);
  (*this)[syllable].push_back(spelling);
  return true;
}

void Script::Merge(const string& s,
                   const SpellingProperties& sp,
                   const vector<Spelling>& v) {
  vector<Spelling>& m((*this)[s]);
  for (const Spelling& x : v) {
    Spelling y(x);
    SpellingProperties& yy(y.properties);
    {
      if (sp.type > yy.type)
        yy.type = sp.type;
      yy.credibility += sp.credibility;
      if (!sp.tips.empty())
        yy.tips = sp.tips;
    }
    auto e = std::find(m.begin(), m.end(), x);
    if (e == m.end()) {
      m.push_back(y);
    } else {
      SpellingProperties& zz(e->properties);
      if (yy.type < zz.type)
        zz.type = yy.type;
      if (yy.credibility > zz.credibility)
        zz.credibility = yy.credibility;
      zz.tips.clear();
    }
  }
}

void Script::Dump(const string& file_name) const {
  std::ofstream out(file_name.c_str());
  for (const value_type& v : *this) {
    bool first = true;
    for (const Spelling& s : v.second) {
      out << (first ? v.first : "") << '\t' << s.str << '\t'
          << "-ac?!"[s.properties.type] << '\t' << s.properties.credibility
          << '\t' << s.properties.tips << std::endl;
      first = false;
    }
  }
  out.close();
}

bool Projection::Load(an<ConfigList> settings) {
  if (!settings)
    return false;
  calculation_.clear();
  Calculus calc;
  bool success = true;
  for (size_t i = 0; i < settings->size(); ++i) {
    an<ConfigValue> v(settings->GetValueAt(i));
    if (!v) {
      LOG(ERROR) << "Error loading formula #" << (i + 1) << ".";
      success = false;
      break;
    }
    const string& formula(v->str());
    an<Calculation> x;
    try {
      x.reset(calc.Parse(formula));
    } catch (boost::regex_error& e) {
      LOG(ERROR) << "Error parsing formula '" << formula << "': " << e.what();
    }
    if (!x) {
      LOG(ERROR) << "Error loading spelling algebra definition #" << (i + 1)
                 << ": '" << formula << "'.";
      success = false;
      break;
    }
    calculation_.push_back(x);
  }
  if (!success) {
    calculation_.clear();
  }
  return success;
}

bool Projection::Apply(string* value) {
  if (!value || value->empty())
    return false;
  bool modified = false;
  Spelling s(*value);
  for (an<Calculation>& x : calculation_) {
    try {
      if (x->Apply(&s))
        modified = true;
    } catch (std::runtime_error& e) {
      LOG(ERROR) << "Error applying calculation: " << e.what();
      return false;
    }
  }
  if (modified)
    value->assign(s.str);
  return modified;
}

// omp_lock_t get_anchor(const string& s) {
//   // static map<string, omp_lock_t> anchors;
//   // static omp_lock_t lock;
//   // omp_set_lock(&lock);
//   // auto e = anchors.find(s);
//   // if (e == anchors.end()) {
//   //   omp_init_lock(&anchors[s]);
//   //   e = anchors.find(s);
//   // }
//   // omp_unset_lock(&lock);
//   // return e->second;

//   return lock;
// }

bool Projection::Apply(Script* value) {
  nvtx3::event_attributes attr{"Projection::Apply", nvtx3::rgb{0, 0, 128}, nvtx3::payload{value->size()}};
  nvtx3::scoped_range r{attr};

  if (!value || value->empty())
    return false;
  bool modified = false;
  int round = 0;

  using IndexedScript = map<string, map<string, SpellingProperties>>;
  IndexedScript indexed_value;
  vector<string> value_keys;

  static omp_lock_t anchors[1024];
  for (int i = 0; i < 1024; ++i) {
    omp_init_lock(&anchors[i]);
  }

  auto get_anchor = [&](const string& s) -> omp_lock_t& {
    size_t h = std::hash<string>{}(s);
    return anchors[h % 1024];
  };

  for (const Script::value_type& v : *value) {
    auto &vs = indexed_value[v.first];
    for (const Spelling& s : v.second) {
      vs[s.str] = s.properties;
    }
    value_keys.push_back(v.first);
  }

  for (an<Calculation>& x : calculation_) {
    ++round;
    LOG(INFO) << "round #" << round;

    nvtx3::event_attributes attr{"Round", nvtx3::rgb{0, 0, 172}, nvtx3::payload{value->size()}};
    nvtx3::scoped_range r{attr};

    IndexedScript new_value;
    vector<string> new_value_keys;
    std::unordered_set<string> new_value_keys_set;
    std::shared_mutex new_value_lock;

    const int num_threads = omp_get_max_threads();
    vector<std::tuple<string, Spelling, bool, string>> applied_spelling_cache[num_threads];

    #pragma omp parallel for
    for (auto &k: value_keys) {
      // phase 1: apply
      auto& v = indexed_value[k];
      Spelling s(k);
      bool applied = false;
      string err;

      try {
        applied = x->Apply(&s);
      } catch (std::runtime_error& e) {
        err = e.what();
        continue;
      }

      applied_spelling_cache[omp_get_thread_num()].emplace_back(k, s, applied, err);
    }

    // phase 2: create entries
    for (int tid = 0; tid < num_threads; tid++) {
      for (auto &t: applied_spelling_cache[tid]){
        auto &[k, s, applied, err] = t;

        if (!err.empty()) {
          LOG(ERROR) << "Error applying calculation: " << err;
          return false;
        }

        modified |= applied;

        if (!applied || (applied && !x->deletion())) {
          // temp.Merge(k, SpellingProperties(), (*value)[k]);
          // Not in new_value, we have to add a new entry
          if (new_value_keys_set.find(k) == new_value_keys_set.end()) {
            new_value.emplace(k, map<string, SpellingProperties>());
            new_value_keys.push_back(k);
            new_value_keys_set.insert(k);
          }
        }

        if (applied && (x->addition() && !s.str.empty())) {
          // temp.Merge(s.str, s.properties, (*value)[k]);
          if (new_value_keys_set.find(s.str) == new_value_keys_set.end()) {
            new_value.emplace(s.str, map<string, SpellingProperties>()); // only create keys, so it is still empty
            new_value_keys.push_back(s.str);
            new_value_keys_set.insert(s.str);
          }
        }
      }
    }

    // phase 3: update entries
    for (int tid = 0; tid < num_threads; tid++) {
      #pragma omp parallel for
      for (auto &t: applied_spelling_cache[tid]){
        auto &[k, s, applied, err] = t;

        if (!applied || (applied && !x->deletion())) {
          // temp.Merge(k, SpellingProperties(), (*value)[k]);
          // entry has been created in new_value during phase 2

          // Now tempering with new_value[k]
          // Get an anchor lock of k
          omp_lock_t anchor = get_anchor(k);
          omp_set_lock(&anchor);

          auto v = indexed_value[k];
          auto it = new_value.find(k);
          auto &vs = it->second;

          for (const auto &[xs, xp]: v) {
            auto e = vs.find(xs);
            if (e == vs.end()) {
              vs[xs] = xp;
            } else {
              SpellingProperties& zz(e->second);
              if (xp.type < zz.type)
                zz.type = xp.type;
              if (xp.credibility > zz.credibility)
                zz.credibility = xp.credibility;
              zz.tips.clear();
            }
          }
          omp_unset_lock(&anchor);
        }

        if (applied && (x->addition() && !s.str.empty())) {
          // temp.Merge(s.str, s.properties, (*value)[k]);
          // entry has been created in new_value during phase 2

          omp_lock_t anchor_s = get_anchor(s.str);
          omp_lock_t anchor_k = get_anchor(k);

          omp_set_lock(&anchor_s);
          omp_set_lock(&anchor_k);

          auto v = indexed_value[k];
          auto it = new_value.find(s.str);
          auto &vs = it->second;

          for (auto &[xs, xp]: v) {
            SpellingProperties& sp = s.properties;
            {
              if (sp.type > xp.type)
                xp.type = sp.type;
              xp.credibility += sp.credibility;
              if (!sp.tips.empty())
                xp.tips = sp.tips;
            }
            auto e = vs.find(xs);
            if (e == vs.end()) {
              vs[xs] = xp;
            } else {
              SpellingProperties& zz(e->second);
              if (xp.type < zz.type)
                zz.type = xp.type;
              if (xp.credibility > zz.credibility)
                zz.credibility = xp.credibility;
              zz.tips.clear();
            }
          }

          omp_unset_lock(&anchor_k);
          omp_unset_lock(&anchor_s);
        }
      }
    }

    indexed_value.swap(new_value);
    value_keys.swap(new_value_keys);
  }

  if (modified) {
    Script temp;
    for (const IndexedScript::value_type& v : indexed_value) {
      vector<Spelling>& m = temp[v.first];
      for (const IndexedScript::mapped_type::value_type& s : v.second) {
        Spelling y(s.first);
        y.properties = s.second;
        m.emplace_back(y);
      }
    }
    value->swap(temp);
  }

  return modified;
}

}  // namespace rime
