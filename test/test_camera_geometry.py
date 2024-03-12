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
        [0.4742,  0.4794,  0.7385],
        [0.1761,  0.7702, -0.6131],
        [-0.8626,  0.4207,  0.2807]
    ])  # Expected rotation matrix (calculated directly = NRT)
    # Calculate actual rotation matrix
    actual = euler_to_rot(yaw, pitch, roll)

    # Assert that actual and expected are close
    assert torch.allclose(actual, expected, rtol=1E-4, atol=1e-4), "Test failed for general case."


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


def test_euler_to_rot_backpropagation():
    # Enable gradient tracking for input tensors
    yaw = torch.tensor([0.1], requires_grad=True)
    pitch = torch.tensor([0.2], requires_grad=True)
    roll = torch.tensor([0.3], requires_grad=True)

    # Define a target rotation matrix as an identity matrix for simplicity
    target = torch.eye(3)

    # Define a simple loss function: Mean Squared Error (MSE) between
    # the generated rotation matrix and the target matrix
    criterion = torch.nn.MSELoss()

    # Forward pass: Compute the rotation matrix
    R = euler_to_rot(yaw, pitch, roll)

    # Compute the loss
    loss = criterion(R, target)

    # Backward pass: Compute gradient of loss w.r.t. inputs
    loss.backward()

    # Check if gradients are not None and have been successfully computed
    assert yaw.grad is not None, "No gradient for yaw"
    assert pitch.grad is not None, "No gradient for pitch"
    assert roll.grad is not None, "No gradient for roll"

    # Optionally, check if gradients are non-zero
    assert yaw.grad.abs().sum().item() > 0, "Zero gradient for yaw"
    assert pitch.grad.abs().sum().item() > 0, "Zero gradient for pitch"
    assert roll.grad.abs().sum().item() > 0, "Zero gradient for roll"

    print("Backpropagation test passed: gradients successfully computed.")


def test_euler_to_rot_gradient_descent():
    # Enable gradient tracking for input tensors
    yaw = torch.tensor([0.1], requires_grad=True)
    pitch = torch.tensor([0.2], requires_grad=True)
    roll = torch.tensor([0.3], requires_grad=True)

    # Define a target rotation matrix as an identity matrix for simplicity
    target = torch.eye(3)

    # Define a simple loss function: Mean Squared Error (MSE) between
    # the generated rotation matrix and the target matrix
    criterion = torch.nn.MSELoss()

    # Define a simple optimizer
    learning_rate = 0.5
    optimizer = torch.optim.SGD([yaw, pitch, roll], lr=learning_rate)

    # Number of optimization steps
    num_steps = 100

    for step in range(num_steps):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Compute the rotation matrix
        R = euler_to_rot(yaw, pitch, roll)

        # Compute the loss
        loss = criterion(R, target)

        # Backward pass: Compute gradient of loss w.r.t. inputs
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Optionally print the loss every few iterations
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    # After optimization, the loss should be significantly reduced
    final_loss = loss.item()
    assert final_loss < 1e-3, "Optimization failed to reduce loss sufficiently"

    print("Gradient descent test passed: loss successfully minimized.")
