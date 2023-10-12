import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QComboBox, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PIL import Image

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.team_logos = {}  # Define the team_logos dictionary here
        self.initUI()
        print(os.getcwd())

    def initUI(self):
        self.setWindowTitle('NFL Match Predictor')
        self.setGeometry(100, 100, 400, 200)

        # Load and set application icon
        app_icon = QIcon(r'C:\Users\Toby\Documents\GitHub\DSP\program\NFL_Logo.jpg')  # Provide the path to your icon file
        self.setWindowIcon(app_icon)

        layout = QGridLayout()

        # Team logo display
        self.away_team_logo = QLabel(self)
        self.home_team_logo = QLabel(self)

        # Labels for dropdown menus
        away_label = QLabel('Away Team:')
        home_label = QLabel('Home Team:')


        # Dropdown menu for away team
        self.away_team_combo = QComboBox()
        self.away_team_combo.addItems(self.team_logos.keys())

        # Dropdown menu for home team
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems(self.team_logos.keys())

        # Construct team logos dictionary
        team_logos_dir = r'C:\Users\Toby\Documents\GitHub\DSP\nfl_teams'
        team_folders = [folder for folder in os.listdir(team_logos_dir) if os.path.isdir(os.path.join(team_logos_dir, folder))]
        self.team_logos = {team: os.path.join(team_logos_dir, team, f'{team.lower()}.png') for team in team_folders}

        # Dropdown menu for home team
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems([team for team in self.team_logos.keys()])

        # Dropdown menu for away team
        self.away_team_combo = QComboBox()
        self.away_team_combo.addItems([team for team in self.team_logos.keys()])

        self.home_team_combo.currentIndexChanged.connect(self.updateLogos)
        self.away_team_combo.currentIndexChanged.connect(self.updateLogos)

        # Predict button
        predict_button = QPushButton('Predict')
        predict_button.clicked.connect(self.predictMatch)

        # Result label
        self.result_label = QLabel('Prediction will be shown here.')

        # Arrange widgets using grid layout
        layout.addWidget(self.away_team_logo, 0, 0, 1, 2)  # Away team logo above dropdown menu (row 0, column 0, rowspan 1, columnspan 2)
        layout.addWidget(away_label, 1, 0)  # Away team label (row 1, column 0)
        layout.addWidget(self.away_team_combo, 1, 1)  # Away team dropdown menu (row 1, column 1)
        layout.addWidget(self.home_team_logo, 2, 0, 1, 2)  # Home team logo above dropdown menu (row 2, column 0, rowspan 1, columnspan 2)
        layout.addWidget(home_label, 3, 0)  # Home team label (row 3, column 0)
        layout.addWidget(self.home_team_combo, 3, 1)  # Home team dropdown menu (row 3, column 1)
        layout.addWidget(predict_button, 4, 0, 1, 2)  # Predict button (row 4, column 0, rowspan 1, columnspan 2)
        layout.addWidget(self.result_label, 5, 0, 1, 2)  # Result label (row 5, column 0, rowspan 1, columnspan 2)

        self.setLayout(layout)

    def updateLogos(self):
        # Get selected teams
        away_team = self.away_team_combo.currentText()
        home_team = self.home_team_combo.currentText()

        # Load and display team logos
        away_logo_path = self.team_logos.get(away_team, '')  # Get the logo path for the away team
        home_logo_path = self.team_logos.get(home_team, '')  # Get the logo path for the home team

        # Set pixmap for away team logo
        if away_logo_path:
            away_pixmap = QPixmap(away_logo_path).scaled(100, 100)
            if not away_pixmap.isNull():
                self.away_team_logo.setPixmap(away_pixmap)
            else:
                print("Error loading away team logo image")
        else:
            self.away_team_logo.clear()  # Clear the pixmap if no logo path is found

        # Set pixmap for home team logo
        if home_logo_path:
            home_pixmap = QPixmap(home_logo_path).scaled(100, 100)
            if not home_pixmap.isNull():
                self.home_team_logo.setPixmap(home_pixmap)
            else:
                print("Error loading home team logo image")
        else:
            self.home_team_logo.clear()  # Clear the pixmap if no logo path is found


    def predictMatch(self):
        # Get selected teams
        away_team = self.away_team_combo.currentText()
        home_team = self.home_team_combo.currentText()

        # Here you can add your prediction logic based on the selected teams
        prediction = f'Predicting {home_team} vs {away_team}... Prediction result goes here.'
        self.result_label.setText(prediction)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
