import serial
import time
import io

def get_package_with_minimum_depth(package_data):
    # Initialize variables to keep track of the minimum depth and the corresponding package
    min_depth = float('inf')
    min_depth_package = None

    for package in package_data:
        if package['class'] == 'package':
            depth = package['depthData']
            if depth < min_depth:
                min_depth = depth
                min_depth_package = package

    return min_depth_package


def send_coordinates_to_serial(x, y, z, label, port="COM6", baudrate=115200, timeout=1):

    # Initialize the serial connection
    serial_inst = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    label = 1 if label else 0
    try:
        # Wrap the serial port with TextIOWrapper~
        ser_io = io.TextIOWrapper(io.BufferedRWPair(serial_inst, serial_inst))

        # Wait for the Arduino to initialize
        time.sleep(2)  # Adjust the delay as needed

        command = f"{str(int(x))},{str(int(y))},{str(int(z))},{str(label)}"
        command = command + "\n"
        print("Sending command:", command)
        serial_inst.write(command.encode("utf-8"))

        print("Waiting for Response:")
        response = ser_io.readline().strip()

        if response:
            try:
                response_text = response
                print("Arduino Response:", response_text)
            except UnicodeDecodeError:
                print("Received non-text data:", response)
        else:
            print("No response received within the timeout.")
    finally:
        # Close the serial connection when done
        serial_inst.close()