# Unscented Kalman Filter Project Starter Code
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project utilize an Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric.

[//]: # (Image References)
[result_dataset1]: output/result_dataset1.png
[result_dataset2]: output/result_dataset2.png

## [Rubric](https://review.udacity.com/#!/rubrics/783/view) Points

### Results

RMSE with dataset1

| Item     | Value       |  
|:--------:|:-----------:|
| X        | 0.0612      |
| Y        | 0.0859      |
| VX       | 0.3302      |
| VY       | 0.2135      |

![Result of dataset1][result_dataset1]

RMSE with dataset2

| Item     | Value       |  
|:--------:|:-----------:|
| X        | 0.0886      |
| Y        | 0.0611      |
| VX       | 0.6583      |
| VY       | 0.2819      |

![Result of dataset2][result_dataset2]

### First measurement
Initialize state vector according to sensor type when receive the first measurement.
```
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

```

### Predict
1) Create augmented sigma points
```
// create augmented state vector
x_aug.head(n_x_) = x_;
x_aug(n_x_) = 0;
x_aug(n_x_ + 1) = 0;

// create augmented covariance matrix
P_aug.fill(0.0);
P_aug.topLeftCorner(n_x_, n_x_) = P_;
P_aug(n_x_, n_x_) = std_a_ * std_a_;
P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

// create square root matrix
MatrixXd L = P_aug.llt().matrixL();

Xsig_aug.col(0)  = x_aug;
for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
}
```

2) Predict sigma points for elapsed time
```
for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    double px_p, py_p;    
    // avoid division by zero
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

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
}

```

3) Predict state mean and state covariance matrix
```
// predict state mean
x.fill(0.0);
for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
}

// predict state covariance matrix
P.fill(0.0);
for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    // angle normalization
    while (x_diff(3) > M_PI)  x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
}
```

### Update
1) Calculate predicted measurement mean and measurement covariance matrix
For Laser type:
```
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
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
}

// add measurement noise covariance matrix
MatrixXd R = MatrixXd(n_z, n_z);
R <<    std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
S = S + R;
```

For Radar type:
```
// create matrix for sigma points in measurement space
MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

// transform sigma points into measurement space
for (int i = 0; i < 2 * n_aug_ + 1; i++) {
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

// predicted measurement mean
VectorXd z_pred = VectorXd(n_z);
z_pred.fill(0.0);
for (int i=0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
}

// measurement covariance matrix S
MatrixXd S = MatrixXd(n_z, n_z);
S.fill(0.0);
for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
}

// add measurement noise covariance matrix
MatrixXd R = MatrixXd(n_z, n_z);
R <<    std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
S = S + R;
```

2) Update state vector and state covariance matrix
```
// create matrix for cross correlation Tc
 MatrixXd Tc = MatrixXd(n_x_, n_z);

 // calculate cross correlation matrix
 Tc.fill(0.0);
 for (int i = 0; i < 2 * n_aug_ + 1; i++) {
     VectorXd z_diff = Zsig_pred.col(i) - z_pred;
     if (n_z == 3) {
         // angle normalization for Radar type
         while (z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
         while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
     }
     // state difference
     VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
     // angle normalization
     while (x_diff(3) > M_PI)  x_diff(3) -= 2. * M_PI;
     while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
 }

 // Kalman gain K
 MatrixXd K = Tc * S.inverse();

 VectorXd z_diff = z - z_pred;
 if (n_z == 3) {
     // angle normalization for Radar type
     while (z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
     while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
 }

 // update state mean and covariance matrix
 x_ = x_pred_ + K * z_diff;
 P_ = P_pred_ - K * S * K.transpose();

```

3) Collect and save NIS (Normalized Innovation Squared) data

```
// Open NIS data files
NIS_radar_.open("../output/NIS_radar.txt", ios::out);
NIS_laser_.open("../output/NIS_laser.txt", ios::out);

...

VectorXd z_diff = z - z_pred;

float NIS_val = z_diff.transpose() * S.inverse() * z_diff;
if (n_z == 3) {
    NIS_radar_ << NIS_val << endl;
} else {
    NIS_laser_ << NIS_val << endl;
}

```
