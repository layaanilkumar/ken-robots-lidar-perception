import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3, Twist


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        self.create_subscription(Float32, '/nearest_obstacle', self.obstacle_callback, 10)
        self.create_subscription(Vector3, '/safe_direction', self.direction_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.nearest_obstacle = None
        self.safe_direction = Vector3()

        self.timer = self.create_timer(0.1, self.control_loop)

    def obstacle_callback(self, msg):
        self.nearest_obstacle = msg.data

    def direction_callback(self, msg):
        self.safe_direction = msg

    def control_loop(self):
        if self.nearest_obstacle is None:
            return

        cmd = Twist()

        if self.nearest_obstacle < 1.0:
            cmd.linear.x = self.safe_direction.x
            cmd.linear.y = self.safe_direction.y
            cmd.angular.z = 0.5
            self.get_logger().info("Obstacle detected → Avoiding")
        else:
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0
            self.get_logger().info("Path clear → Moving forward")

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()