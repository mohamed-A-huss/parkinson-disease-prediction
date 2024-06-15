import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# Function to load the disease detection model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to make predictions using the loaded model
def predict(model, features):
    
    # Perform any preprocessing needed on 'features'
    # For example, convert to numpy array
    print(features)
    features_array = np.array(features).reshape(1, -1)
    print(features_array)
    #features_array = scaler.fit_transform(features_array)
    scaler_path = "C:/Users/Babak/Desktop/study/CODECLAUSE/DS/P3/scaler.pkl"
    scaler = load_model(scaler_path)
    features_scaled = scaler.transform(features_array)
    print(features_scaled)
    
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return prediction

class DiseaseDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Parkinson\'s Disease Detection')
        self.setGeometry(100, 100, 400, 300)

        # Create labels, line edits, and button
        self.label1 = QLabel('MDVP:Fo(Hz):', self)
        self.lineEdit1 = QLineEdit(self)

        self.label2 = QLabel('MDVP:Fhi(Hz):', self)
        self.lineEdit2 = QLineEdit(self)

        self.label3 = QLabel('MDVP:Flo(Hz):', self)
        self.lineEdit3 = QLineEdit(self)
        
        self.label4 = QLabel('MDVP:Jitter(Abs):', self)
        self.lineEdit4 = QLineEdit(self)

        self.label5 = QLabel('MDVP:Shimmer:', self)
        self.lineEdit5 = QLineEdit(self)

        self.label6 = QLabel('NHR:', self)
        self.lineEdit6 = QLineEdit(self)

        self.label7 = QLabel('HNR:', self)
        self.lineEdit7 = QLineEdit(self)

        self.label8 = QLabel('RPDE:', self)
        self.lineEdit8 = QLineEdit(self)

        self.label9 = QLabel('DFA:', self)
        self.lineEdit9 = QLineEdit(self)

        self.label10 = QLabel('spread1:', self)
        self.lineEdit10 = QLineEdit(self)

        self.label11 = QLabel('spread2:', self)
        self.lineEdit11 = QLineEdit(self)

        self.label12 = QLabel('D2:', self)
        self.lineEdit12 = QLineEdit(self)

        self.predictButton = QPushButton('Predict', self)
        self.predictButton.clicked.connect(self.predictClicked)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.lineEdit1)
        layout.addWidget(self.label2)
        layout.addWidget(self.lineEdit2)
        layout.addWidget(self.label3)
        layout.addWidget(self.lineEdit3)
        layout.addWidget(self.label4)
        layout.addWidget(self.lineEdit4)
        layout.addWidget(self.label5)
        layout.addWidget(self.lineEdit5)
        layout.addWidget(self.label6)
        layout.addWidget(self.lineEdit6)
        layout.addWidget(self.label7)
        layout.addWidget(self.lineEdit7)
        layout.addWidget(self.label8)
        layout.addWidget(self.lineEdit8)
        layout.addWidget(self.label9)
        layout.addWidget(self.lineEdit9)
        layout.addWidget(self.label10)
        layout.addWidget(self.lineEdit10)
        layout.addWidget(self.label11)
        layout.addWidget(self.lineEdit11)
        layout.addWidget(self.label12)
        layout.addWidget(self.lineEdit12)
        layout.addWidget(self.predictButton)
        self.setLayout(layout)

        self.model_path = "C:/Users/Babak/Desktop/study/CODECLAUSE/DS/P3/pyqt/catboost_model.pkl"
        self.model = load_model(self.model_path)

    def predictClicked(self):
        # Get input values
        feature1 = float(self.lineEdit1.text())
        feature2 = float(self.lineEdit2.text())
        feature3 = float(self.lineEdit3.text())
        feature4 = float(self.lineEdit4.text())
        feature5 = float(self.lineEdit5.text())
        feature6 = float(self.lineEdit6.text())
        feature7 = float(self.lineEdit7.text())
        feature8 = float(self.lineEdit8.text())
        feature9 = float(self.lineEdit9.text())
        feature10 = float(self.lineEdit10.text())
        feature11 = float(self.lineEdit11.text())
        feature12 = float(self.lineEdit12.text())
        # Add more features as needed
        
        # Make prediction
        features = [feature1, feature2, feature3,
                    feature4,feature5,feature6,
                    feature7,feature8,feature9,
                    feature10,feature11,feature12]  # Convert to appropriate type
        prediction = predict(self.model, features)
        
        # Determine prediction text
        if prediction == 1:
            result = 'Parkinson\'s Disease detected.'
        else:
            result = 'No Parkinson\'s Disease detected.'
        
        # Show prediction result in a message box
        QMessageBox.information(self, 'Prediction Result', result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DiseaseDetectorApp()
    window.show()
    sys.exit(app.exec_())
