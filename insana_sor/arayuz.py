import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QTextEdit,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QDesktopWidget,
)
from PyQt5.QtCore import Qt


class QuestionAnswerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        # Merkezde başlat
        self.center()

        # Load CSV file
        self.df = pd.read_csv("insana_sorulacak_veri_cevapli.csv")
        self.index = 0

        # Widgets
        self.question_label = QLabel("", self)
        self.question_edit = QTextEdit(self)
        self.question_edit.setReadOnly(True)
        self.question_edit.setLineWrapMode(QTextEdit.WidgetWidth)  # Line wrap mode

        self.answer_labels = [
            QLabel("Trendyol Mistral:", self),
            QLabel("Sambalingo Llama2:", self),
            QLabel("Meta Llama3:", self),
            QLabel("Cosmos Llama3:", self),
        ]
        self.answer_edits = [QTextEdit(self) for _ in range(4)]

        for edit in self.answer_edits:
            edit.setReadOnly(True)
            edit.setLineWrapMode(QTextEdit.WidgetWidth)  # Line wrap mode

        self.show_button = QPushButton("Göster", self)
        self.hide_button = QPushButton("Gizle", self)
        self.prev_button = QPushButton("Geri", self)
        self.next_button = QPushButton("İleri", self)
        self.info_label = QLabel("", self)  # Label for question index information

        # Connect buttons
        self.show_button.clicked.connect(self.show_answers)
        self.hide_button.clicked.connect(self.hide_answers)
        self.prev_button.clicked.connect(self.prev_question)
        self.next_button.clicked.connect(self.next_question)

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.show_button)
        button_layout.addWidget(self.hide_button)

        # Answer Layout
        answer_layout = QGridLayout()
        answer_layout.addWidget(self.answer_labels[0], 0, 0)
        answer_layout.addWidget(self.answer_edits[0], 0, 1)
        answer_layout.addWidget(self.answer_labels[1], 0, 2)
        answer_layout.addWidget(self.answer_edits[1], 0, 3)
        answer_layout.addWidget(self.answer_labels[2], 1, 0)
        answer_layout.addWidget(self.answer_edits[2], 1, 1)
        answer_layout.addWidget(self.answer_labels[3], 1, 2)
        answer_layout.addWidget(self.answer_edits[3], 1, 3)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.question_label)
        main_layout.addWidget(self.question_edit)
        main_layout.addLayout(answer_layout)
        main_layout.addWidget(self.info_label)  # Add info label to the main layout
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.update_question()

        # Initialize button visibility
        self.show_button.setVisible(True)
        self.hide_button.setVisible(False)

        self.setWindowTitle("Question Answer App")
        self.show()

    def center(self):
        # Get screen geometry
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_question(self):
        row = self.df.iloc[self.index]
        self.question_label.setText("Soru:")
        self.question_edit.setPlainText(row["question"])
        self.answer_edits[0].setPlainText(row["trendyol_mistral"])
        self.answer_edits[1].setPlainText(row["sambalingo_llama2"])
        self.answer_edits[2].setPlainText(row["meta_llama3"])
        self.answer_edits[3].setPlainText(row["cosmos_llama3"])
        self.hide_answers()  # Start with answers hidden
        info_text = f"Soru {self.index + 1} / {len(self.df)}\tIndex: {row['index']}"
        # Update the information label with current question index
        self.info_label.setText(info_text)

    def show_answers(self):
        for label in self.answer_labels:
            label.setVisible(True)
        self.show_button.setVisible(False)
        self.hide_button.setVisible(True)

    def hide_answers(self):
        for label in self.answer_labels:
            label.setVisible(False)
        self.show_button.setVisible(True)
        self.hide_button.setVisible(False)

    def prev_question(self):
        if self.index > 0:
            self.index -= 1
            self.update_question()

    def next_question(self):
        if self.index < len(self.df) - 1:
            self.index += 1
            self.update_question()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = QuestionAnswerApp()
    sys.exit(app.exec_())
