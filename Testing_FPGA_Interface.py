
import numpy as np
import serial
import struct
import time

def get_serial_port(serial_port_number=''):
    import platform
    import subprocess
    serial_port_number = str(serial_port_number)
    SERIAL_PORT = None
    try:
        system = platform.system()
        if system == 'Darwin':  # Mac
            SERIAL_PORT = subprocess.check_output('ls -a /dev/tty.usbserial*', shell=True).decode("utf-8").strip()  # Probably '/dev/tty.usbserial-110'
        elif system == 'Linux':
            SERIAL_PORT = '/dev/ttyUSB' + serial_port_number  # You might need to change the USB number
        elif system == 'Windows':
            SERIAL_PORT = 'COM' + serial_port_number
        else:
            raise NotImplementedError('For system={} connection to serial port is not implemented.')
    except Exception as err:
        print(err)

    return SERIAL_PORT




PING_TIMEOUT            = 1.0       # Seconds
CALIBRATE_TIMEOUT       = 10.0      # Seconds
READ_STATE_TIMEOUT      = 1.0      # Seconds
SERIAL_SOF              = 0xAA
CMD_PING                = 0xC0

class Interface:
    def __init__(self):
        self.device         = None
        self.msg            = []
        self.start = None
        self.end = None

        self.encoderDirection = None

    def open(self, port, baud):
        self.port = port
        self.baud = baud
        self.device = serial.Serial(port, baudrate=baud, timeout=None)
        self.device.reset_input_buffer()

    def close(self):
        if self.device:
            time.sleep(2)
            self.device.close()
            self.device = None

    def clear_read_buffer(self):
        self.device.reset_input_buffer()

    def ping(self):
        msg = [SERIAL_SOF, CMD_PING, 4]
        msg.append(self._crc(msg))
        self.device.write(bytearray(msg))
        self.device.flush()
        return self._receive_reply(CMD_PING, 4, PING_TIMEOUT) == msg

    def send_net_input(self, net_input):
        self.device.reset_output_buffer()
        # self.clear_read_buffer()
        # msg = [SERIAL_SOF, net_input]
        # msg.append(self._crc(msg))
        bytes_written = self.device.write(bytearray(net_input))
        print(bytes_written)
        self.device.flush()

    def receive_net_output(self):
        self.clear_read_buffer()
        # reply = self._receive_reply(4, READ_STATE_TIMEOUT)
        net_output = self.device.read(size=4)
        # net_output = struct.unpack('=3hBIH', bytes(net_output[3:16]))
        net_output = struct.unpack('<f', net_output)
        # net_output=reply
        return net_output

    def _receive_reply(self, cmdLen, timeout=None, crc=True):
        self.device.timeout = timeout
        self.start = False

        while True:
            c = self.device.read()
            # Timeout: reopen device, start stream, reset msg and try again
            if len(c) == 0:
                print('\nReconnecting.')
                self.device.close()
                self.device = serial.Serial(self.port, baudrate=self.baud, timeout=timeout)
                self.clear_read_buffer()
                time.sleep(1)
                self.msg = []
                self.start = False
            else:
                self.msg.append(ord(c))
                if self.start == False:
                    self.start = time.time()

            while len(self.msg) >= cmdLen:
                # print('I am looping! Hurra!')
                # Message must start with SOF character
                if self.msg[0] != SERIAL_SOF:
                    #print('\nMissed SERIAL_SOF')
                    del self.msg[0]
                    continue

                # Check message packet length
                if self.msg[2] != cmdLen and cmdLen < 256:
                    print('\nWrong Packet Length.')
                    del self.msg[0]
                    continue

                # Verify integrity of message
                if crc and self.msg[cmdLen-1] != self._crc(self.msg[:cmdLen-1]):
                    print('\nCRC Failed.')
                    del self.msg[0]
                    continue

                self.device.timeout = None
                reply = self.msg[:cmdLen]
                del self.msg[:cmdLen]
                return reply

    def _crc(self, msg):
        crc8 = 0x00

        for i in range(len(msg)):
            val = msg[i]
            for b in range(8):
                sum = (crc8 ^ val) & 0x01
                crc8 >>= 1
                if sum > 0:
                    crc8 ^= 0x8C
                val >>= 1

        return crc8




my_fake_data = [-2.449621446430683136e-02,
                -9.724658727645874023e-01,
                2.330452054738998413e-01,
                7.049140520393848419e-03,
                1.333827376365661621e-01,
                1.000000000000000000e+00,
                0.000000000000000000e+00]
what_I_expect_to_get =[7.742919921875000000e-01]

my_fake_data = np.array(my_fake_data, dtype=np.float32)
what_I_expect_to_get = np.array(what_I_expect_to_get)

SERIAL_PORT = get_serial_port(serial_port_number=3)
SERIAL_BAUD = 115200

InterfaceInstance = Interface()
InterfaceInstance.open(SERIAL_PORT, SERIAL_BAUD)

InterfaceInstance.send_net_input(my_fake_data)
net_output = InterfaceInstance.receive_net_output()
print(net_output)

# difference = np.abs(net_output-what_I_expect_to_get)