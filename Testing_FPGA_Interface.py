from Control_Toolkit.Controllers.controller_fpga import Interface, get_serial_port
import numpy as np

my_fake_data = [-2.449621446430683136e-02,
                -9.724658727645874023e-01,
                2.330452054738998413e-01,
                7.049140520393848419e-03,
                1.333827376365661621e-01,
                1.000000000000000000e+00,
                0.000000000000000000e+00]
what_I_expect_to_get =[7.742919921875000000e-01]

my_fake_data = np.array(my_fake_data)
what_I_expect_to_get = np.array(what_I_expect_to_get)

SERIAL_PORT = get_serial_port(serial_port_number=1)
SERIAL_BAUD = 230400

InterfaceInstance = Interface()
InterfaceInstance.open(SERIAL_PORT, SERIAL_BAUD)

InterfaceInstance.send_net_input(my_fake_data)
net_output = InterfaceInstance.receive_net_output()

difference = np.abs(net_output-what_I_expect_to_get)