#include<ros/ros.h>
#include<string>
#include<iostream>
#include<nav_msgs/Odometry.h>
#include<nav_msgs/Path.h>
#include<geometry_msgs/Pose.h>
#include<geometry_msgs/PoseStamped.h>
#include<ros/time.h>

double lx, ly, distance;
nav_msgs::Path path;
ros::Publisher path_pub;
ros::Time start_time;
bool first_time = true;

void poseCallback(const nav_msgs::Odometry &odom){
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "odom";

    geometry_msgs::PoseStamped this_pose_stamped;
    this_pose_stamped.pose =  odom.pose.pose;
    this_pose_stamped.header.stamp = ros::Time::now();
    this_pose_stamped.header.frame_id = "odom";

    if (first_time)
    {
        start_time =  ros::Time::now();
        lx = this_pose_stamped.pose.position.x;
        ly = this_pose_stamped.pose.position.y;
        distance = sqrt(pow((lx),2)+pow((ly),2));
        first_time = false;
    }else{
        distance += sqrt(pow((this_pose_stamped.pose.position.x-lx),2)+pow((this_pose_stamped.pose.position.y-ly),2));
        lx = this_pose_stamped.pose.position.x;
        ly = this_pose_stamped.pose.position.y;
    }
    double last_time = this_pose_stamped.header.stamp.toSec() - start_time.toSec();
    ROS_INFO("time:%f , length:%f",last_time,distance);
    

    path.poses.push_back(this_pose_stamped);
    path_pub.publish(path);
}
int main(int argc,char** argv){

    ros::init(argc, argv, "show_trajectory");
    ros::NodeHandle nh;
    start_time =  ros::Time::now();
    std::string trajTopic,odomTopic;
    nh.param<std::string>("trajTopic", trajTopic, "/NF_odom2path");
    nh.param<std::string>("odomTopic", odomTopic, "/wheel_odom");
    path_pub = nh.advertise<nav_msgs::Path>(trajTopic, 1, true);
    ros::Subscriber pose_sub = nh.subscribe(odomTopic, 10, poseCallback);
    ros::Rate loop_rate(10);

    while(ros::ok()){
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
