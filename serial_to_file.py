import serial
import argparse

def serial_to_file(port: str, baudrate: int, output_file: str):
    ser = serial.Serial(port, baudrate)
    
    try:
        with open(output_file, 'a') as output_file_handle:
            while True:
                line = ser.readline()
                decoded_line = line.decode('utf-8').strip()
                output_file_handle.write(decoded_line + '\n')
    except KeyboardInterrupt:
        print("Monitoring stopped.")
    finally:
        ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor a serial port and write output to a file.')
    parser.add_argument('port', type=str, help='Serial port name (e.g., COM3 or /dev/ttyS3)')
    parser.add_argument('baudrate', type=int, help='Baudrate (e.g., 9600)')
    parser.add_argument('output_file', type=str, help='Output file name')
    args = parser.parse_args()

    print(f"Start monitoring on port {args.port} with baudrate {args.baudrate}")
    serial_to_file(args.port, args.baudrate, args.output_file)