    def predictMatch(self):
            # Get selected teams
            away_team = self.away_team_combo.currentText()
            home_team = self.home_team_combo.currentText()

            # Load and display team logos
            self.away_team_logo.setPixmap(QPixmap(self.team_logos[away_team]).scaled(100, 100))  # Set logo size as needed
            self.home_team_logo.setPixmap(QPixmap(self.team_logos[home_team]).scaled(100, 100))  # Set logo size as needed

            # Here you can add your prediction logic based on the selected teams
            prediction = f'Predicting {home_team} vs {away_team}... Prediction result goes here.'
            self.result_label.setText(prediction)