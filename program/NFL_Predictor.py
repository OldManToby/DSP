import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QComboBox, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QIcon

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('NFL Match Predictor')
        self.setGeometry(100, 100, 400, 200)

       # Load and set application icon
        app_icon = QIcon(r'C:\Users\Toby\Documents\GitHub\DSP\program\NFL_Logo.jpg')  # Provide the path to your icon file
        self.setWindowIcon(app_icon)

        layout = QGridLayout()

        # Labels
        home_label = QLabel('Home Team:')
        away_label = QLabel('Away Team:')

        # Load and display team logos
        team_logos = [f for f in os.listdir() if f.endswith('.png')]

        # Dropdown menu for home team
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems([os.path.splitext(logo)[0] for logo in team_logos])

        # Dropdown menu for away team
        self.away_team_combo = QComboBox()
        self.away_team_combo.addItems([os.path.splitext(logo)[0] for logo in team_logos])

        # Predict button
        predict_button = QPushButton('Predict')
        predict_button.clicked.connect(self.predictMatch)

        # Result label
        self.result_label = QLabel('Prediction will be shown here.')

         # Arrange widgets in the grid layout
        layout.addWidget(self.away_team_combo, 0, 0, 1, 1)  # Away team on the left (row 0, column 0, rowspan 1, columnspan 1)
        layout.addWidget(self.home_team_combo, 0, 2, 1, 1)  # Home team on the right (row 0, column 2, rowspan 1, columnspan 1)
        layout.addWidget(predict_button, 1, 1, 1, 1)  # Predict button in the middle below the menus (row 1, column 1, rowspan 1, columnspan 1)
        layout.addWidget(self.result_label, 2, 0, 1, 3)  # Result label spanning 1 row and 3 columns

        self.setLayout(layout)

    def predictMatch(self):
        # Here you can add your prediction logic based on the selected teams
        home_team = self.home_team_combo.currentText()
        away_team = self.away_team_combo.currentText()
        prediction = f'Predicting {home_team} vs {away_team}... Prediction result goes here.'
        self.result_label.setText(prediction)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
