import sys
import os
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QComboBox, QPushButton, QLabel, QStyleFactory, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from model import train_and_predict
from team_names import team_names

class DisclaimerDialog(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Disclaimer')
        self.setText("This tool is for research purposes only and should not be used for any form of gambling.\n\nThe data used a presented is provided by The Football Database(www.footballdb.com) & contains statistics from 2000 onwards.\n"
                      "\nThe accuracy of this model is not guaranteed.\n\n"
                      "By clicking 'Acknowledge,' you agree to use this tool responsibly.")
        self.setIcon(QMessageBox.Information)
        self.addButton(QPushButton('Acknowledge'), QMessageBox.AcceptRole)

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.team_logos = {}
        self.available_teams = list(self.team_logos.keys())
        self.initUI()

    def initUI(self):
        disclaimer_dialog = DisclaimerDialog()
        disclaimer_dialog.exec_()
        self.setWindowTitle('NFL Match Predictor')
        self.setGeometry(100,100,600,450)
        app_icon = QIcon('NFL_Logo.jpg')
        self.setWindowIcon(app_icon)
        layout = QGridLayout()

        self.away_team_logo = QLabel(self)
        self.home_team_logo = QLabel(self)

        away_label = QLabel('Away Team:')
        home_label = QLabel('Home Team:')

        # Construct team logos dictionary using team_names keys
        team_logos_dir = 'nfl_teams'
        for folder_name, team_name in team_names.items():
            logo_path = os.path.join(team_logos_dir, folder_name, f'{folder_name.lower()}.png')
            self.team_logos[team_name] = logo_path

        # Populate available teams
        self.available_teams = list(self.team_logos.keys())

        # Dropdown menu for home/away team
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems(self.available_teams)

        self.away_team_combo = QComboBox()
        self.away_team_combo.addItems(self.available_teams)

        self.home_team_combo.currentIndexChanged.connect(self.updateLogos)
        self.away_team_combo.currentIndexChanged.connect(self.updateLogos)

        self.home_team_combo.setCurrentText(random.choice(self.available_teams))
        self.away_team_combo.setCurrentText(random.choice(self.available_teams))

        self.updateLogos()

        predict_button = QPushButton('Predict')
        predict_button.clicked.connect(self.on_predict_button_clicked)

        self.result_label = QLabel('Prediction will be shown here.')
        self.result_label.setWordWrap(True)
        self.result_label.setFixedSize(300, 60)

        layout.addWidget(self.away_team_logo,0,0,2,2)
        layout.addWidget(self.home_team_logo,0,2,2,2)

        layout.addWidget(away_label,2,0)
        layout.addWidget(self.away_team_combo,2,1)
        layout.addWidget(home_label,2,2)
        layout.addWidget(self.home_team_combo,2,3)
        layout.addWidget(predict_button,3,1,1,2)
        layout.addWidget(self.result_label,4,0,2,2)

        self.setLayout(layout)

    def updateLogos(self):
        away_team = self.away_team_combo.currentText()
        home_team = self.home_team_combo.currentText()

        away_logo_path = self.team_logos.get(away_team, '')
        home_logo_path = self.team_logos.get(home_team, '')

        if away_logo_path:
            away_pixmap = QPixmap(away_logo_path).scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            if not away_pixmap.isNull():
                self.away_team_logo.setPixmap(away_pixmap)
                self.away_team_logo.setStyleSheet("border: 2px solid black;")  # Set border style
            else:
                print("Error loading away team logo image")
        else:
            self.away_team_logo.clear()
        if home_logo_path:
            home_pixmap = QPixmap(home_logo_path).scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            if not home_pixmap.isNull():
                self.home_team_logo.setPixmap(home_pixmap)
                self.home_team_logo.setStyleSheet("border: 2px solid black;")
            else:
                print("Error loading home team logo image")
        else:
            self.home_team_logo.clear()

        # Disable selected teams in the other dropdown menu
        selected_home_team = self.home_team_combo.currentText()
        selected_away_team = self.away_team_combo.currentText()

        self.away_team_combo.blockSignals(True)
        self.away_team_combo.clear()
        self.away_team_combo.addItems([team for team in self.available_teams if team != selected_home_team])
        self.away_team_combo.setCurrentText(selected_away_team)
        self.away_team_combo.blockSignals(False)

        self.home_team_combo.blockSignals(True)
        self.home_team_combo.clear()
        self.home_team_combo.addItems([team for team in self.available_teams if team != selected_away_team])
        self.home_team_combo.setCurrentText(selected_home_team)
        self.home_team_combo.blockSignals(False)

    def on_predict_button_clicked(self):
        team1_selection = self.home_team_combo.currentText()
        team2_selection = self.away_team_combo.currentText()
        # Call the function once and store the result
        summary = train_and_predict(team1_selection, team2_selection)
        # Update the GUI with the result
        self.result_label.setText(summary)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())