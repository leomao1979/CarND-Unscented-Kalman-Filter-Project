#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;
    
    // initial state vector
    x_ = VectorXd(5);
    
    // initial covariance matrix
    P_ = MatrixXd::Identity(5, 5);
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    
    is_initialized_ = false;
    n_x_ = x_.size();
    n_aug_ = n_x_ + 2;
    lambda_ = 3 - n_aug_;
    
    weights_ = VectorXd(2 * n_aug_ + 1);
    // set weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }
    
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    x_pred_ = VectorXd(n_x_);
    P_pred_ = MatrixXd(n_x_, n_x_);
    
    // Open NIS data files
    NIS_radar_.open("../output/NIS_radar.txt", ios::out);
    NIS_laser_.open("../output/NIS_laser.txt", ios::out);
    // Check for errors opening the files
    if (!NIS_radar_.is_open()) {
        cout << "Error opening NIS_radar.txt" << endl;
        exit(1);
    }
    
    if (!NIS_laser_.is_open()) {
        cout << "Error opening NIS_laser.txt" << endl;
        exit(1);
    }
}

UKF::~UKF() {
    NIS_laser_.close();
    NIS_radar_.close();
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        cout << "UKF: " << endl;
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates and initialize state.
            float rho     = meas_package.raw_measurements_(0);
            float theta   = meas_package.raw_measurements_(1);
            float px = rho * cos(theta);
            float py = rho * sin(theta);
            x_ << px, py, 0, 0, 0;
        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
        }
        
        // Done initializing, no need to predict or update
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;  //dt - expressed in seconds
    time_us_ = meas_package.timestamp_;
    
    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    }
    
    // print the output
//    cout << "x_ = " << endl << x_ << endl;
//    cout << "P_ = " << endl << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    AugmentedSigmaPoints(Xsig_aug);
    SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance();
}

void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_out) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    
    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    
    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
 
    //create augmented mean state
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0;
    x_aug(n_x_ + 1) = 0;
    
    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
    
    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    
    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }
    
    //write result
    Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, double delta_t) {
    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);
        
        //predicted state values
        double px_p, py_p;
        
        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        } else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }
        
        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;
        
        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;
        
        //write predicted sigma point into right column
        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }
    
    //write result
    Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
    //create vector for predicted state
    VectorXd x = VectorXd(n_x_);
    
    //create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);

    //predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        x = x + weights_(i) * Xsig_pred_.col(i);
    }
    
    //predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;
        //angle normalization
        while (x_diff(3) > M_PI)  x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        
        P = P + weights_(i) * x_diff * x_diff.transpose() ;
    }

    x_pred_ = x;
    P_pred_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 2;
    //create matrix for sigma points in measurement space
    MatrixXd Zsig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);
    // predicted measurement mean
    VectorXd z_pred = VectorXd(n_z);
    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    PredictLidarMeasurement(Zsig_pred, z_pred, S);
    
    //create example vector for incoming radar measurement
    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);
    
    UpdateState(z, Zsig_pred, z_pred, S);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;
    //create matrix for sigma points in measurement space
    MatrixXd Zsig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);
    // predicted measurement mean
    VectorXd z_pred = VectorXd(n_z);
    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    PredictRadarMeasurement(Zsig_pred, z_pred, S);
    
    //create example vector for incoming radar measurement
    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);
    
    UpdateState(z, Zsig_pred, z_pred, S);
 }

void UKF::PredictLidarMeasurement(MatrixXd& Zsig_out, VectorXd& z_out, MatrixXd& S_out) {
    // measurement dimension. radar can measure r, phi, and r_dot
    int n_z = z_out.size();
    
    // matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig = Xsig_pred_.block(0, 0, n_z, 2 * n_aug_ + 1);

    // predicted measurement mean
    VectorXd z_pred = VectorXd(n_z);
    z_pred = x_pred_.head(n_z);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R <<    std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;
    S = S + R;

    Zsig_out = Zsig;
    z_out = z_pred;
    S_out = S;
}

void UKF::PredictRadarMeasurement(MatrixXd& Zsig_out, VectorXd& z_out, MatrixXd& S_out) {
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = z_out.size();
    
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        
        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;
        
        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        while (z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R <<    std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
    S = S + R;
    
    //write result
    Zsig_out = Zsig;
    z_out = z_pred;
    S_out = S;
}

void UKF::UpdateState(const VectorXd& z, const MatrixXd& Zsig_pred, const VectorXd& z_pred, const MatrixXd& S) {
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = z.size();
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig_pred.col(i) - z_pred;
        if (n_z == 3) {
            //angle normalization
            while (z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
            while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
        }
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
        //angle normalization
        while (x_diff(3) > M_PI)  x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //residual
    VectorXd z_diff = z - z_pred;
    if (n_z == 3) {
        //angle normalization
        while (z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    }
    float NIS_val = z_diff.transpose() * S.inverse() * z_diff;
    if (n_z == 3) {
        NIS_radar_ << NIS_val << endl;
    } else {
        NIS_laser_ << NIS_val << endl;
    }

    //update state mean and covariance matrix
    x_ = x_pred_ + K * z_diff;
    P_ = P_pred_ - K * S * K.transpose();
}
