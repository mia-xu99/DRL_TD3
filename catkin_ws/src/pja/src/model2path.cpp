#include<ros/ros.h>
#include<string>
#include<iostream>
#include<nav_msgs/Odometry.h>
#include<nav_msgs/Path.h>
#include<geometry_msgs/Pose.h>
#include<geometry_msgs/PoseStamped.h>
#include<gazebo_msgs/GetModelState.h>

double ix, iy, px, py;
nav_msgs::Path path;
ros::Publisher path_pub;

void poseCallback(const nav_msgs::Odometry &odom){
//    px = p_msg.pose.pose.position.x;
//    py = p_msg.pose.pose.position.y;

    //打印运动轨迹

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "/odom";

    geometry_msgs::PoseStamped this_pose_stamped;
    this_pose_stamped.pose =  odom.pose.pose;
//    this_pose_stamped.pose = py-iy;

//    this_pose_stamped.pose.orientation.x = 0;
//    this_pose_stamped.pose.orientation.y = 0;
//    this_pose_stamped.pose.orientation.z = 0;
//    this_pose_stamped.pose.orientation.w = 1;
    this_pose_stamped.header.stamp = ros::Time::now();
    this_pose_stamped.header.frame_id = "/odom";

//    path.poses.push_back(this_pose_stamped);
//    path_pub.publish(path);
}
int main(int argc,char** argv){

    ros::init(argc, argv, "show_trajectory");
    ros::NodeHandle nh;
    std::string trajTopic,odomTopic;
    nh.param<std::string>("trajTopic", trajTopic, "/trajectory");
    nh.param<std::string>("odomTopic", odomTopic, "/odom");

    ros::ServiceClient client = nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    path_pub = nh.advertise<nav_msgs::Path>(trajTopic, 1, true);
//    ros::Subscriber pose_sub = nh.subscribe(odomTopic, 10, poseCallback);
    bool is_start = true;
    ros::Rate loop_rate(10);

    while(ros::ok()){
        gazebo_msgs::GetModelState modelState;
        modelState.request.model_name = ("robot");
        modelState.request.relative_entity_name = ("world");
        client.call(modelState);

        if (modelState.response.success) {
            geometry_msgs::PoseStamped this_pose_stamped;
            this_pose_stamped.pose = modelState.response.pose;
            this_pose_stamped.header.stamp = ros::Time::now();
            this_pose_stamped.header.frame_id = "odom";

            path.poses.push_back(this_pose_stamped);
            path.header.frame_id = "odom";
            path.header.stamp = ros::Time::now();
            path_pub.publish(path);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}