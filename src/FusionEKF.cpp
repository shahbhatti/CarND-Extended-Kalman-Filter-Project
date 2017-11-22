#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices

  // Measurement Noise Covariance matrix R for Laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  // Measurement Noise Covariance matrix R for Radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // Measurement Matrix H for Laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1,0,0,0,
        0, 1, 0, 0;

  //Hj_ = MatrixXd(3, 4);
  //Hj will be initialized when we calculate the Jacobian

  // State Covariance matrix P (Prediction Error)
  MatrixXd P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  //  State Transition Matrix F - will be initialized during ProcessMeasurement
  MatrixXd F_ = MatrixXd(4, 4);

  // Process Noise Covariance matrix Q- will be initialized during ProcessMeasurement
  MatrixXd Q_ = MatrixXd(4, 4);

  // State Vector x
  VectorXd x_ = VectorXd(4);
  x_ << 1, 1, 1, 1;

  // Initialize the Kalman Filter class
  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement

    cout << "Initialization in EKF" << endl;

    double px = 0;
    double py = 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        //polar to catesian conversion before initialization
        double rho = measurement_pack.raw_measurements_[0];
        double phi = measurement_pack.raw_measurements_[1];
        double rho_dot = measurement_pack.raw_measurements_[2];

        px = rho * cos(phi);
        py = rho * sin(phi);

        ekf_.x_ << px, py, rho_dot * cos(phi), rho_dot * sin(phi);

        if(fabs(px) < .00001){
            px = 1;
            ekf_.P_(0,0)=1000;
        }

        if(fabs(py) < .00001){
            py = 1;
            ekf_.P_(1,1)=1000;

        }

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        //initialization
        px = measurement_pack.raw_measurements_[0];
        py = measurement_pack.raw_measurements_[1];

        ekf_.x_ << px, py, 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  double current_timestamp = measurement_pack.timestamp_;
  double time_diff = (current_timestamp - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ << 1, 0, time_diff, 0,
          0, 1, 0, time_diff,
          0, 0, 1, 0,
          0, 0, 0, 1;

  double dt_4 = time_diff * time_diff * time_diff * time_diff;
  double dt_3 = time_diff * time_diff * time_diff;
  double dt_2 = time_diff * time_diff;

  double noise_ax = 9;
  double noise_ay = 9;

  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
          0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
          dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
          0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    Tools tools;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
