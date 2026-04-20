#!/usr/bin/env python3
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys
import termios
import tty
import select

class JointKeyboardControl(Node):
    def __init__(self):
        super().__init__('joint_keyboard_control')
        self.publisher_ = self.create_publisher(JointState, '/joint_command', 10)

        # 关节名称（与 /joint_states 一致）
        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper'
        ]

        # 当前关节位置（初始化为0）
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 当前选中的关节索引（默认第一个关节）
        self.selected_joint = 0

        # 关节调整步长（每次按键变化的幅度）
        self.step_size = 0.01

        # 保存终端设置（用于恢复）
        self.old_settings = termios.tcgetattr(sys.stdin)

        # 提示用户如何操作
        self.print_instructions()

    def print_instructions(self):
        print("\n=== 机械臂键盘控制 ===")
        print("1-6: 选择关节")
        print("a/d: 调整关节位置（a=减小, d=增大）")
        print("q: 退出程序")
        print("当前关节:", self.joint_names[self.selected_joint])
        print("当前位置:", self.joint_positions[self.selected_joint])

    def get_key(self):
        # 非阻塞读取键盘输入
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def run(self):
        try:
            # 设置终端为非阻塞模式
            tty.setcbreak(sys.stdin.fileno())

            while rclpy.ok():
                key = self.get_key()
                if key is not None:
                    # 选择关节（1-6）
                    if key in ['1', '2', '3', '4', '5', '6']:
                        self.selected_joint = int(key) - 1
                        print("\n当前关节:", self.joint_names[self.selected_joint])
                        print("当前位置:", self.joint_positions[self.selected_joint])

                    # 调整关节位置（a=减小, d=增大）
                    elif key == 'a':
                        self.joint_positions[self.selected_joint] -= self.step_size
                        self.publish_joint_command()
                        print("位置 -:", self.joint_positions[self.selected_joint])

                    elif key == 'd':
                        self.joint_positions[self.selected_joint] += self.step_size
                        self.publish_joint_command()
                        print("位置 +:", self.joint_positions[self.selected_joint])

                    # 退出程序（q）
                    elif key == 'q':
                        print("\n退出程序...")
                        break

                # 短暂休眠，避免CPU占用过高
                rclpy.spin_once(self, timeout_sec=0.01)

        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def publish_joint_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    joint_keyboard_control = JointKeyboardControl()

    try:
        joint_keyboard_control.run()
    except KeyboardInterrupt:
        pass
    finally:
        joint_keyboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()