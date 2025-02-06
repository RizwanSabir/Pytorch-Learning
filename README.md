# PyTorch Implementation

This repository contains PyTorch implementations for various neural network models, including:

- **Linear Classification** (Binary & Multi-class Classification)
- **Convolutional Neural Networks (CNNs)**

## 📌 Features

- Implementations using **PyTorch**
- Support for **binary and multi-class classification**
- CNN architecture for more complex tasks
- Easy-to-follow structure for training and evaluation



## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.7+** installed along with the following dependencies:

```bash
pip install torch torchvision numpy matplotlib
```


## 🔍 How `Flatten` Layer Input Size is Determined
When defining the fully connected (`nn.Linear`) layer in a CNN, the number of input features must be calculated after the image has been processed through the convolutional and pooling layers.

### Automatic Calculation of Flattened Size
Instead of manually calculating the input size for `nn.Linear`, we use a helper method `calculate_flattened_size()`:

```python
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.calculate_flattened_size(input_shape), out_features=output_shape)
        )
    
    def calculate_flattened_size(self, input_shape: int) -> int:
        """
        Passes a dummy tensor through the conv layers to determine the final feature map size.
        """
        dummy_input = torch.zeros(1, input_shape, 32, 32)  # Assuming input images are 32x32
        dummy_output = self.conv_block_2(self.conv_block_1(dummy_input))
        return dummy_output.view(1, -1).size(1)
```

### Explanation
- A **dummy input tensor** is created with shape `(1, input_channels, height, width)`.
- It is passed through the convolutional and pooling layers to get the final feature map.
- The output is flattened and the total number of elements is calculated dynamically.
- This ensures that the correct number of features is passed to the `nn.Linear` layer.




## 📊 Results

The models provide performance metrics such as accuracy and loss, which are logged during training.

## 🛠 Future Improvements

- Add support for **Recurrent Neural Networks (RNNs)**
- Implement **Hyperparameter tuning**
- Improve dataset augmentation techniques

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the repository!

## 📜 License

This project is licensed under the **MIT License**.

---

🔥 Happy Coding with PyTorch! 🚀


