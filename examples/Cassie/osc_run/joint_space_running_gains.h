#include "systems/controllers/osc/osc_gains.h"
#include "yaml-cpp/yaml.h"

#include "drake/common/yaml/yaml_read_archive.h"

using Eigen::MatrixXd;

struct JointSpaceRunningGains : OSCGains {
  std::vector<double> JointW;
  std::vector<double> JointKp;
  std::vector<double> JointKd;

  template <typename Archive>
  void Serialize(Archive* a) {
    OSCGains::Serialize(a);
    a->Visit(DRAKE_NVP(w_input));
    a->Visit(DRAKE_NVP(w_accel));
    a->Visit(DRAKE_NVP(w_soft_constraint));
    a->Visit(DRAKE_NVP(impact_threshold));
    a->Visit(DRAKE_NVP(mu));
    a->Visit(DRAKE_NVP(JointW));
    a->Visit(DRAKE_NVP(JointKp));
    a->Visit(DRAKE_NVP(JointKd));
  }
};