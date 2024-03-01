import torch
from pixr.camera.camera_geometry import euler_to_rot


def test_euler_to_rot_eye():
    # Zero rotation
    R = euler_to_rot(torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]))
    expected_R = torch.eye(3)
    assert torch.allclose(R, expected_R), "Test case failed for identity matrix"


def test_euler_to_rot_basic():
    # Define test angles in radians for a general case
    yaw = torch.tensor([1.0])
    pitch = torch.tensor([0.5])
    roll = torch.tensor([-0.5])

    expected = torch.tensor([
        [0.4742,  0.7748,  0.4182],
        [-0.2590,  0.5767, -0.7748],
        [-0.8415,  0.2590,  0.4742]
    ])  # Expected rotation matrix (calculated directly = NRT)
    # Calculate actual rotation matrix
    actual = euler_to_rot(yaw, pitch, roll)

    # Assert that actual and expected are close
    assert torch.allclose(actual, expected, rtol=1E-4, atol=1e-5), "Test failed for general case."


def test_euler_to_rot_pure_pitch():
    # Define a pitch angle and set yaw and roll to zero
    pitch = torch.tensor([0.5])  # 28. degrees
    yaw = torch.tensor([0.0])
    roll = torch.tensor([0.0])

    # Compute the rotation matrix
    R = euler_to_rot(yaw, pitch, roll)

    # Expected rotation matrix for pure X rotation
    expected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(pitch).item(), -torch.sin(pitch).item()],
        [0.0, torch.sin(pitch).item(), torch.cos(pitch).item()]
    ])

    # Assert the actual and expected matrices are close
    assert torch.allclose(R, expected), "Test failed for pure pitch rotation."


def test_euler_to_rot_orthonormality():
    # Example Euler angles (in radians)
    yaw = torch.tensor([0.1])
    pitch = torch.tensor([0.2])
    roll = torch.tensor([0.3])

    # Generate the rotation matrix
    R = euler_to_rot(yaw, pitch, roll)

    # Calculate the product of the rotation matrix and its transpose
    R_transpose = R.t()  # Transpose of R
    identity_matrix = torch.eye(3)  # 3x3 Identity matrix

    # Check if the product of R and its transpose is close to the identity matrix
    assert torch.allclose(torch.mm(R, R_transpose), identity_matrix, atol=1e-6), "Matrix is not orthonormal"

    # Additionally, to explicitly check the determinant is +1 (optional)
    assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-6), "Determinant of rotation matrix is not 1"
