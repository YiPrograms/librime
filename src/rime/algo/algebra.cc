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

bool Projection::Apply(Script* value) {
  nvtx3::event_attributes attr{"Projection::Apply", nvtx3::rgb{0, 0, 128}, nvtx3::payload{value->size()}};
  nvtx3::scoped_range r{attr};

  if (!value || value->empty())
    return false;

  const int n_threads = omp_get_max_threads();
  const int keys_per_thread = value->size() / n_threads;

  map<string, vector<Spelling>>::const_iterator thread_start[n_threads + 1];

  auto it = value->begin();
  for (int i = 0; i < n_threads; ++i) {
    thread_start[i] = it;

    if (i == n_threads - 1)
      thread_start[i + 1] = value->end();
    else
      std::advance(it, keys_per_thread);
  }

  Script *thread_script[n_threads];
  bool modified = false;

  bool error = false;

  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
  
    Script *local_script = reinterpret_cast<Script *>(
      new map<string, vector<Spelling>>(thread_start[thread_id], thread_start[thread_id + 1])
    );

    thread_script[thread_id] = local_script;

    int round = 0;

    for (an<Calculation>& x : calculation_) {
      if (error)
        break;

      ++round;
      DLOG(INFO) << "round #" << round;

      nvtx3::event_attributes attr{"Round", nvtx3::rgb{0, 0, 172}, nvtx3::payload{local_script->size()}};
      nvtx3::scoped_range r{attr};

      Script new_script;
      for (const Script::value_type& v : *local_script) {
        Spelling s(v.first);
        bool applied = false;
        try {
          applied = x->Apply(&s);
        } catch (std::runtime_error& e) {
          LOG(ERROR) << "Error applying calculation: " << e.what();
          error = true;
        }
        if (applied) {
          modified = true;
          if (!x->deletion()) {
            new_script.Merge(v.first, SpellingProperties(), v.second);
          }
          if (x->addition() && !s.str.empty()) {
            new_script.Merge(s.str, s.properties, v.second);
          }
        } else {
          new_script.Merge(v.first, SpellingProperties(), v.second);
        }
      }
      local_script->swap(new_script);
    }

  } // end omp parallel

  if (error)
    return false;

  if (modified) {
    // Merge all thread scripts

    Script new_value;
    for (int i = 0; i < n_threads; ++i) {
      for (const Script::value_type& v : *thread_script[i]) {
        new_value.Merge(v.first, SpellingProperties(), v.second);
      }
    }
    value->swap(new_value);
  }

  LOG(INFO) << "Total Size: " << value->size();
  
  return modified;
}

}  // namespace rime
