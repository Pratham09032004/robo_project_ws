#include <string>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"


class PS4_ROS2: public rclcpp::Node
{
    public:
        PS4_ROS2(): Node("ps4_ros2"){
            this->declare_parameter("scale_linear", 1.0);
            this->declare_parameter("scale_angular", 1.0);
            this->declare_parameter<std::string>("pub_topic", "/roboteq_controller/cmd_vel_unstamped");
            this->declare_parameter<std::string>("emergency_topic", "/emergency");

            this->get_parameter("scale_linear", scale_linear);
            this->get_parameter("scale_angular", scale_angular);
            this->get_parameter("pub_topic", pub_name);
            this->get_parameter("emergency_stop", emergency_name);

            sub = this->create_subscription<sensor_msgs::msg::Joy>("/joy", 10, std::bind(&PS4_ROS2::subscribePS4, this, std::placeholders::_1));

            chat = this->create_publisher<geometry_msgs::msg::Twist>(pub_name, 1000);
            emergency_stop = this->create_publisher<std_msgs::msg::Bool>("/emergency", 1000);

            calib1 = 0;
            calib2 = 0;
            calib = false;

            maxVel = scale_linear;
            maxVelR = scale_linear * -1;

            RCLCPP_INFO(this->get_logger(), "maxVelR: %f", maxVelR);
            RCLCPP_INFO(this->get_logger(), "scale_linear set to: %f", scale_linear);
            RCLCPP_INFO(this->get_logger(), "scale_angular set to: %f", scale_angular);
            RCLCPP_INFO(this->get_logger(), "PS4_ROS2 initialized");


        }

        ~PS4_ROS2(){}

        void run() {
            if(calib) {
                publishTwistMsg();
                emergency();
            }
        }
    
        void prepareData()
        {
            // Normalize velocity between ]-1.0, 1.0[
            send_l2 = (-0.5 * l2) + 0.5;
            send_r2 = (r2 - 1.0) * 0.5;
    
            // Apply rosparam "scale_linear"
            send_l2 = scale_linear * send_l2;
            send_r2 = scale_linear * send_r2;
    
            // Apply rosparam "scale_angular"
            send_leftStickX = scale_angular * leftStickX;
        }
    
        void publishTwistMsg() {
            geometry_msgs::msg::Twist msg;
            msg.linear.x = 0.0;
            msg.linear.y = 0.0;
            msg.linear.z = 0.0;
    
            msg.angular.x = 0.0;
            msg.angular.y = 0.0;
            msg.angular.z = 0.0;
    
            prepareData();
    
            //printRaw();
            if(!buttonTouch) {
                 if ((buttonX==true) && (send_l2 >= 0.01) && (send_l2 <= maxVel)){
                   msg.linear.x = send_l2;
    
                } else if ((buttonX==true) && (send_r2 <= 0.0) && (send_r2 >= maxVelR)) {
                    msg.linear.x = send_r2;
    
                }   else if ((buttonSq==true) && (send_l2 >= 0.01) && (send_l2 <= maxVel)){
                    msg.linear.x = send_l2/5;
            
                }  else if ((buttonSq==true) && (send_r2 <= 0.0) && (send_r2 >= maxVelR)){
                    msg.linear.x = send_r2/5;
    
                } else if ((send_l2 >= 0.01) && (send_l2 <= maxVel)) {
                    msg.linear.x = send_l2/10;
    
                } else if ((send_r2 <= 0.0) && (send_r2 >= maxVelR)) {
                    msg.linear.x = send_r2/10;
    
                } 
                
                // Drehbewegung
                msg.angular.z = send_leftStickX / 10;
            }
             
    
            
            else{
                //ROS_WARN("SENDING EMERGENCY STOP");
    
                /* To Do */
            }
    
            chat->publish(msg);
        }

        void emergency()
        {
            std_msgs::msg::Bool emergency_msg;
            emergency_msg.data = false; 

            if (buttonO == 1) 
            {
                emergency_msg.data = true;

            }
            else 
            {
                emergency_msg.data = false;
            }


            emergency_stop->publish(emergency_msg);

        }

        void subscribePS4(const sensor_msgs::msg::Joy::SharedPtr joy) {

            // ROS2 controls

            buttonSq = joy->buttons[3];
            buttonX = joy->buttons[0];
            buttonO = joy->buttons[1];
            buttonTr = joy->buttons[2];
            //buttonTouch = joy->buttons[13];
            l1 = joy->buttons[4];
            r1 = joy->buttons[5];
    
            l2 = joy->axes[2];
            r2 = joy->axes[5];
    
            leftStickX = joy->axes[0];
            leftStickY = joy->axes[1];
            rightStickX = joy->axes[3];
            rightStickY = joy->axes[4]; 
    
        }
    
        bool calibrate()
        {
            double progress = ((double) calib1 / calib_duration) * 100;
            if( (l2 == -1.0) && (r2 == -1.0) )
            {
                RCLCPP_INFO(this->get_logger(),"Press L2 and R2 to calibrate: %i ", (int) progress);
                calib1++;
                calib2++;
            }
            else{
                calib1 = 0;
                calib2 = 0;
            }

            if( (calib1 > calib_duration) && (calib2 > calib_duration))
            {
                calib = true;
                return true;
            }
            else
                return false;
                RCLCPP_INFO(this->get_logger(),"calib1: %i , calib2: %i ", (int)calib1, (int)calib2);
            
        }

        bool waitForRelease()
        {
            if( (l2 == 1.0) && (r2 == 1.0) )
            {
                return true;
            }
            else{
                return false;
            }
        }

        void printSend()
        {

            RCLCPP_INFO(this->get_logger(),"#####################################");
            RCLCPP_INFO(this->get_logger(),"Send L2: %f", send_l2);
            RCLCPP_INFO(this->get_logger(),"Send R2: %f", send_r2);
            RCLCPP_INFO(this->get_logger(),"Touch Button pressed: %i", buttonTouch);
            RCLCPP_INFO(this->get_logger(),"Left Stick Y: %f", leftStickY);
            RCLCPP_INFO(this->get_logger(),"Left Stick X: %f", leftStickX);
            RCLCPP_INFO(this->get_logger(),"##################################### \n");
        }

        void printRaw()
        {

            RCLCPP_INFO(this->get_logger(),"#####################################");
            RCLCPP_INFO(this->get_logger(),"Squared Button pressed: %i", buttonSq);
            RCLCPP_INFO(this->get_logger(),"X Button pressed: %i", buttonX);
            RCLCPP_INFO(this->get_logger(),"O Button pressed: %i", buttonO);
            RCLCPP_INFO(this->get_logger(),"Triangel Button pressed: %i", buttonTr);
            RCLCPP_INFO(this->get_logger(),"Left/Right Button pressed: %i", arrowsX);
            RCLCPP_INFO(this->get_logger(),"Down/Up Button pressed: %i", arrowsY);
            RCLCPP_INFO(this->get_logger(),"Touch Button pressed: %i", buttonTouch);
            RCLCPP_INFO(this->get_logger(),"L1: %i", l1);
            RCLCPP_INFO(this->get_logger(),"R1: %i", r1);
            RCLCPP_INFO(this->get_logger(),"L2: %f", l2);
            RCLCPP_INFO(this->get_logger(),"R2: %f", r2);
            RCLCPP_INFO(this->get_logger(),"Left Stick Y: %f", leftStickY);
            RCLCPP_INFO(this->get_logger(),"Left Stick X: %f", leftStickX);
            RCLCPP_INFO(this->get_logger(),"Right Stick Y: %f", rightStickY);
            RCLCPP_INFO(this->get_logger(),"Right Stick X: %f", rightStickX);
            RCLCPP_INFO(this->get_logger(),"##################################### \n");
        }


    private:
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr chat;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr emergency_stop;
        rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub;

        /* calibration variables */
        int calib_duration = 20; // 1/10 sec
        int calib1, calib2;
        bool calib;

        /* raw data */
        double leftStickY, leftStickX, rightStickY, rightStickX, l2, r2;
        int arrowsX, arrowsY, buttonSq, buttonO, buttonTr,buttonTouch, l1, r1;
        //int buttonX;
        bool buttonX = false;

        /* rosparams */
        double scale_linear, scale_angular;
        std::string pub_name;
        std::string emergency_name;

        double maxVel, maxVelR;

        /* send data */
        double send_leftStickX, send_l2, send_r2;
};


int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    PS4_ROS2 *ps4_ros2_raw = new PS4_ROS2;
    std::shared_ptr<PS4_ROS2> ps4_ros2(ps4_ros2_raw);

    // calibrate
    std::cout<<"Press L2 and R2 to calibrate"<<std::endl;
    RCLCPP_INFO(ps4_ros2->get_logger(), "Press L2 and R2 to calibrate");

    bool ready =  false;

    while(!ready){
        ready = ps4_ros2->calibrate();
        rclcpp::spin_some(ps4_ros2);
        rclcpp::sleep_for(std::chrono::milliseconds(100));
    }

    RCLCPP_INFO(ps4_ros2->get_logger(), "Release L2 and R2");
    rclcpp::sleep_for(std::chrono::milliseconds(2000));
    ready =  false;

    while(!ready){
        ready = ps4_ros2->waitForRelease();
        rclcpp::spin_some(ps4_ros2);
        rclcpp::sleep_for(std::chrono::milliseconds(100));
    }

    RCLCPP_INFO(ps4_ros2->get_logger(), "Calibrated - ready to use");

    rclcpp::Rate loop_rate(10);

    while(rclcpp::ok()){
        ps4_ros2->run();
        rclcpp::spin_some(ps4_ros2);
        loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
    
}