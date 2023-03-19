from unittest import TestCase

from xor import train_model, \
    evaluate_model, \
    XORLinear, \
    XORNonLinear


class XORTest(TestCase):
    def test_linear_nn(self):
        xor_linear_model = XORLinear()
        xor_linear_model = train_model(xor_linear_model)
        accuracy = evaluate_model(xor_linear_model)
        self.assertTrue(accuracy == 50)

    def test_nonlinear_nn(self):
        xor_nonlinear_model = XORNonLinear()
        xor_nonlinear_model = train_model(xor_nonlinear_model)
        accuracy = evaluate_model(xor_nonlinear_model)
        self.assertTrue(accuracy == 100)
