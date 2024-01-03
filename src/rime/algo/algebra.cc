//
// Copyright RIME Developers
// Distributed under the BSD License
//
// 2012-01-19 GONG Chen <chen.sst@gmail.com>
//
#include <algorithm>
#include <fstream>
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

omp_lock_t get_anchor(const string& s) {
  // static map<string, omp_lock_t> anchors;
  static omp_lock_t lock;
  // omp_set_lock(&lock);
  // auto e = anchors.find(s);
  // if (e == anchors.end()) {
  //   omp_init_lock(&anchors[s]);
  //   e = anchors.find(s);
  // }
  // omp_unset_lock(&lock);
  // return e->second;

  return lock;
}

bool Projection::Apply(Script* value) {
  nvtx3::event_attributes attr{"Projection::Apply", nvtx3::rgb{0, 0, 128}, nvtx3::payload{value->size()}};
  nvtx3::scoped_range r{attr};

  if (!value || value->empty())
    return false;
  bool modified = false;
  int round = 0;

  using IndexedScript = map<string, map<string, SpellingProperties>>;
  IndexedScript indexed_value;

  for (const Script::value_type& v : *value) {
    for (const Spelling& s : v.second) {
      indexed_value[s.str][v.first] = s.properties;
    }
  }

  for (an<Calculation>& x : calculation_) {
    ++round;
    DLOG(INFO) << "round #" << round;

    nvtx3::event_attributes attr{"Round", nvtx3::rgb{0, 0, 172}, nvtx3::payload{value->size()}};
    nvtx3::scoped_range r{attr};

    IndexedScript temp;
    vector<string> scriptKeys;

    for (const IndexedScript::value_type& v : indexed_value) {
      scriptKeys.push_back(v.first);
    }

    #pragma omp parallel for
    for (auto &k: scriptKeys) {
      auto &v = indexed_value[k];
      Spelling s(k);
      bool applied = false;
      string err;
      try {
        applied = x->Apply(&s);
      } catch (std::runtime_error& e) {
        err = e.what();
        continue;
      }
      if (applied) {
          modified = true;
          if (!x->deletion()) {
            // temp.Merge(k, SpellingProperties(), (*value)[k]);

            omp_lock_t anchor = get_anchor(k);
            omp_set_lock(&anchor);

            auto &vs = temp[k];
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
          if (x->addition() && !s.str.empty()) {
            // temp.Merge(s.str, s.properties, (*value)[k]);

            omp_lock_t anchor = get_anchor(s.str);
            omp_set_lock(&anchor);

            auto &vs = temp[s.str];
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

            omp_unset_lock(&anchor);
          }
      } else {
        // temp.Merge(k, SpellingProperties(), (*value)[k]);

        omp_lock_t anchor = get_anchor(k);
        omp_set_lock(&anchor);

        auto &vs = temp[k];
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

      // const int thread_id = omp_get_thread_num();
      // appliedSpellingCache[thread_id].emplace_back(k, s, applied, err);
    }

    indexed_value.swap(temp);
  }

  if (modified) {
    Script temp;
    for (const IndexedScript::value_type& v : indexed_value) {
      for (const IndexedScript::mapped_type::value_type& s : v.second) {
        Spelling sp(v.first);
        sp.properties = s.second;
        temp[v.second.begin()->first].emplace_back(sp);
      }
    }
    value->swap(temp);
  }

  return modified;
}

}  // namespace rime
