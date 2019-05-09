# Experimental Setups



### CTNN Settings

Train Acc** = 0.919  **Test Acc** = 0.907

Layer: 2

Thresholds: 15, 10

Data dropout: 0.2

receptive field: (7,7), (7,7)

Layer One Channels: 30

Layer Two Channels: 100

DoG filters: (1,2), (2,1), ON, OFF

Epoch: 1

Winner-Take-all: (8,5), (8, 5)

Timestamp: 30



| Classifier                      | Test Accuracy |
| ------------------------------- | ------------- |
| KNN                             | 82.59%        |
| Decision Tree                   | 65.71%        |
| Random Forests                  | 77.16         |
| AdaBoost                        | 63.93%        |
| GradientBoost                   | 83.84%        |
| Gauissian NB                    | 54.18%        |
| Linear Discriminant Analysis    | 83.61%        |
| Quadratic Discriminant Analysis | 86.70%        |
| SVM                             | **90.7%**     |
| One-layer Neural Nework         | **92.60%**    |

