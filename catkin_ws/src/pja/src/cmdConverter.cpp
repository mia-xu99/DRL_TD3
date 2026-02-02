#include <math.h>
#include <ros/ros.h>

#include <message_filters/synchronizer.h>

#include <std_msgs/Int8.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>

#include <tf/transform_datatypes.h>


using namespace std;

const double PI = 3.1415926;

double sensorOffsetX = 0;
double sensorOffsetY = 0;
int pubSkipNum = 1;
int pubSkipCount = 0;
bool twoWayDrive = true;
double lookAheadDis = 0.5;
double yawRateGain = 7.5;
double stopYawRateGain = 7.5;
double maxYawRate = 45.0;
double maxSpeed = 1.0;
double maxAccel = 1.0;
double switchTimeThre = 1.0;
double dirDiffThre = 0.1;
double stopDisThre = 0.2;
double slowDwnDisThre = 1.0;
bool useInclRateToSlow = false;
double inclRateThre = 120.0;
double slowRate1 = 0.25;
double slowRate2 = 0.5;
double slowTime1 = 2.0;
double slowTime2 = 2.0;
bool useInclToStop = false;
double inclThre = 45.0;
double stopTime = 5.0;
bool noRotAtStop = false;
bool noRotAtGoal = true;
bool autonomyMode = false;
double autonomySpeed = 1.0;
double joyToSpeedDelay = 2.0;

float joySpeed = 0;
float joySpeedRaw = 0;
float joyYaw = 0;
int safetyStop = 0;

float vehicleX = 0;
float vehicleY = 0;
float vehicleZ = 0;
float vehicleRoll = 0;
float vehiclePitch = 0;
float vehicleYaw = 0;

float vehicleXRec = 0;
float vehicleYRec = 0;
float vehicleZRec = 0;
float vehicleRollRec = 0;
float vehiclePitchRec = 0;
float vehicleYawRec = 0;

float vehicleYawRate = 0;
float vehicleSpeed = 0;

double odomTime = 0;
double joyTime = 0;
double slowInitTime = 0;
double stopInitTime = false;
int pathPointID = 0;
bool pathInit = false;
bool navFwd = true;
double switchTime = 0;

nav_msgs::Path path;
ros::Subscriber subCmd;
ros::Subscriber subOdom;
ros::Publisher pubSpeed;
geometry_msgs::TwistStamped cmd_vel;


void cmdHandler(const geometry_msgs::Twist::ConstPtr& cmd)
{
    pubSkipCount--;
    if (pubSkipCount < 0) {
        cmd_vel.header.stamp = ros::Time().fromSec(odomTime);
//        if (fabs(vehicleSpeed) <= maxAccel / 100.0) cmd_vel.twist.linear.x = 0;
//        else cmd_vel.twist.linear.x = cmd;
//        cmd_vel.twist.angular.z = vehicleYawRate;
        cmd_vel.twist=*cmd;
        pubSpeed.publish(cmd_vel);

        pubSkipCount = pubSkipNum;
    }
}


void odomHandler(const nav_msgs::Odometry::ConstPtr& odomIn)
{
    odomTime = odomIn->header.stamp.toSec();

    double roll, pitch, yaw;
    geometry_msgs::Quaternion geoQuat = odomIn->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

    vehicleRoll = roll;
    vehiclePitch = pitch;
    vehicleYaw = yaw;
    vehicleX = odomIn->pose.pose.position.x - cos(yaw) * sensorOffsetX + sin(yaw) * sensorOffsetY;
    vehicleY = odomIn->pose.pose.position.y - sin(yaw) * sensorOffsetX - cos(yaw) * sensorOffsetY;
    vehicleZ = odomIn->pose.pose.position.z;

    if ((fabs(roll) > inclThre * PI / 180.0 || fabs(pitch) > inclThre * PI / 180.0) && useInclToStop) {
        stopInitTime = odomIn->header.stamp.toSec();
    }

    if ((fabs(odomIn->twist.twist.angular.x) > inclRateThre * PI / 180.0 || fabs(odomIn->twist.twist.angular.y) > inclRateThre * PI / 180.0) && useInclRateToSlow) {
        slowInitTime = odomIn->header.stamp.toSec();
    }
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "cmdConverter");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");


   subCmd = nh.subscribe<geometry_msgs::Twist> ("/move_base/cmd_vel", 5, cmdHandler);
   subOdom = nh.subscribe<nav_msgs::Odometry> ("/state_estimation", 5, odomHandler);


  pubSpeed = nh.advertise<geometry_msgs::TwistStamped> ("/cmd_vel", 5);
  cmd_vel.header.frame_id = "vehicle";


  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
