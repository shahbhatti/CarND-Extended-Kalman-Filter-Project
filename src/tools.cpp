#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  // Calculate the root mean sqaure error (rmse)

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() == 0 || estimations.size() !=  ground_truth.size()) {
    cout <<  "Invalid estimation or ground truth" << endl;
    return rmse;
  }

  for (int i=0; i < estimations.size(); ++i){
    VectorXd res = estimations[i]- ground_truth[i];
    res = res.array()* res.array();
    rmse += res;
  }

  // mean
  rmse = rmse/estimations.size();

  // square root
  rmse = rmse.array().sqrt();

  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  // calculate the Jacobian

  // Initialize
  MatrixXd Hj(3, 4);
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  // precompute values
  double c1 = px*px+py*py;
  double c2 = sqrt(c1);
  double c3 = c1*c2;

  // check division by 0
  if (fabs(c1) < 0.0001) {
    cout << "Division by 0 while calculating Jacobian" << endl;
    return Hj;
  }

  // compute Jacobian
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;

}
