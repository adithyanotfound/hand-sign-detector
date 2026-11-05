"""Build the project report PDF. Needs: pip install reportlab==4.2.5 pillow (newer reportlab needs Python 3.10+)."""
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
OUT = os.path.join(ROOT, "Hand_Sign_Detection_Report.pdf")

FONTS = "/System/Library/Fonts/Supplemental/Times New Roman"
for name, suffix in (("TNR", ""), ("TNR-B", " Bold"), ("TNR-I", " Italic"), ("TNR-BI", " Bold Italic")):
    pdfmetrics.registerFont(TTFont(name, f"{FONTS}{suffix}.ttf"))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

WIDTH = letter[0] - 2 * inch
BODY = ParagraphStyle("body", fontName="TNR", fontSize=12, leading=14, alignment=TA_JUSTIFY, spaceAfter=4)
ITEM = ParagraphStyle("item", parent=BODY, spaceAfter=0)
TITLE = ParagraphStyle("title", fontName="TNR-B", fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=8)
NAME = ParagraphStyle("name", fontName="TNR-B", fontSize=14, leading=17, alignment=TA_CENTER)
HEAD = ParagraphStyle("head", fontName="TNR-B", fontSize=14, leading=17, spaceBefore=7, spaceAfter=2, keepWithNext=1)
CAPTION = ParagraphStyle("caption", fontName="TNR-I", fontSize=12, leading=14, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6)
CELL = ParagraphStyle("cell", fontName="TNR", fontSize=12, leading=14)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="TNR-B")


def p(text):
    return Paragraph(text, BODY)


def head(text):
    return Paragraph(text, HEAD)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, ITEM), leftIndent=26, bulletColor=colors.black) for t in items],
                        bulletType="bullet", start="•", bulletFontName="TNR", bulletFontSize=12, leftIndent=26)


def numbered(items):
    return ListFlowable([ListItem(Paragraph(t, ITEM), leftIndent=26) for t in items], bulletType="1",
                        bulletFontName="TNR", bulletFontSize=12, leftIndent=26, bulletFormat="%s.")


def image(name, width):
    w, h = PILImage.open(os.path.join(FIG, name)).size
    return Image(os.path.join(FIG, name), width=width, height=width * h / w)


def figure(name, caption, width=WIDTH):
    return KeepTogether([image(name, width), Paragraph(caption, CAPTION)])


def table(rows, widths, caption):
    data = [[Paragraph(str(c), CELL_B if i == 0 else CELL) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return KeepTogether([Paragraph(caption, CAPTION), t, Spacer(1, 6)])


story = [
    Paragraph("<u>PRCV PROJECT (COMPUTER VISION BASED)</u>", TITLE),
    Paragraph("Real-Time Hand Sign Detection", NAME),
    Paragraph("Lightweight Landmark-Based Recognition from a Webcam", NAME),

    head("Aim:"),
    p("Real-time detection of hand signs from a laptop webcam using MediaPipe hand landmarks and a small "
      "neural network trained on a public dataset."),
    head("Objective:"),
    p("To recognise five common signs (Hello, I Love You, Peace, Thank You and Yes) on a MacBook without any "
      "cloud API, to show the tracked hand and the confidence of each sign live, and to measure accuracy on "
      "people the model was not trained on."),

    head("Theory:"),
    p("<b>Dataset:</b> Sign_Language_Dataset has 1,770 webcam-style images from about three people (A, B and C), "
      "in six gesture folders. Its “ok” folder shows two raised fingers rather than the usual OK gesture, so those "
      "images were used as “no sign” examples. SignAlphaSet supplied further “no sign” examples: 40 images "
      "each of 18 alphabet letters, leaving out letters that overlap the five signs."),
    p("<b>Models Compared:</b>"),
    numbered([
        "Logistic Regression on x, y, z landmarks",
        "Neural network with one hidden layer of 32 units on x, y, z landmarks",
        "Neural network with hidden layers of 64 and 32 units on x, y, z landmarks",
        "Neural network with hidden layers of 64 and 32 units on x, y landmarks (selected)",
    ]),
    Spacer(1, 4),
    bullets([
        "MediaPipe Hand Landmarker is a pretrained model that finds a hand and returns 21 keypoints.",
        "Normalisation moves the wrist to the origin and divides by the hand’s size, so the features do not "
        "depend on where the hand is or how far it is from the camera.",
        "The selected network maps 42 inputs to six outputs (five signs and “no sign”). It has 5,030 "
        "parameters and takes 23 KB on disk.",
        "Temperature scaling (T = 1.9) softens overconfident outputs. A sign is named only when its smoothed "
        "confidence reaches 60%.",
    ]),
    Spacer(1, 4),
    p("<b>Evaluation Metrics:</b> Macro-F1, per-sign recall, false-accept rate, confusion matrix, processing "
      "time per frame, CPU use and memory. Testing is leave-one-person-out: the images are burst photos, so a "
      "random split would put near-identical frames in both training and test sets and inflate the score."),

    head("Methodology"),
    numbered([
        "Install MediaPipe, OpenCV, NumPy and scikit-learn in a virtual environment.",
        "Download both datasets and verify their published checksums.",
        "Extract hand landmarks from each image (hand found in 1,766 of 1,770 gesture images).",
        "Group images by person, using clothing colour clusters as an approximate identity.",
        "Normalise the landmarks and build the feature vectors.",
        "Augment training data with mirroring, rotation up to 20° and small landmark noise, and balance "
        "the classes by resampling.",
        "Compare the four models with leave-one-person-out testing over five random seeds.",
        "Set the temperature and confidence threshold from the held-out results.",
        "Train the selected model on all 2,476 usable images and export its weights for NumPy.",
        "Check the final model on images of the letter V, which it never trained on.",
        "Build the real-time application with OpenCV and test it on video files.",
    ]),
    Spacer(1, 6),
    figure("fig1_pipeline.png", "Fig. 1: Real-time pipeline. Only the neural net (blue) was trained in this project."),
    figure("fig2_signs.png", "Fig. 2: The five signs and a “no sign” example, with landmarks. Images cropped and "
           "annotated from the two datasets (CC BY 4.0).", WIDTH),

    head("Real-Time Application Interface"),
    p("The application is an OpenCV desktop window that runs entirely on the laptop. It acts as the "
      "presentation layer between the webcam and the trained model. The interface allows users to:"),
    bullets([
        "See the tracked hand skeleton and bounding box on a mirrored camera view.",
        "Read the recognised sign and its confidence next to the hand.",
        "Compare live confidence bars for all five signs and “no sign”.",
        "Build a line of text from signs held for 0.6 seconds, and clear it with a key press.",
        "Monitor the frame rate and the processing time per frame.",
    ]),

    head("Output:"),
    table([
        ["Model", "Macro-F1", "Person A", "Person B", "Person C"],
        ["Logistic Regression", "0.910 ± 0.007", "99.0%", "71.1%", "98.4%"],
        ["Net (32), x, y, z", "0.902 ± 0.023", "98.2%", "66.0%", "99.6%"],
        ["Net (64, 32), x, y, z", "0.907 ± 0.037", "97.9%", "68.3%", "99.6%"],
        ["<b>Net (64, 32), x, y</b>", "<b>0.958 ± 0.006</b>", "<b>100%</b>", "<b>81.6%</b>", "<b>100%</b>"],
    ], [1.9 * inch, 1.35 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch],
        "Table 1: Model comparison on held-out people: macro-F1 and accuracy per person, averaged over "
        "five seeds, before the 60% threshold."),
    figure("fig3_app.jpg", "Fig. 3: Application output on a test video built from dataset images, some of which "
           "were used in training. This shows the interface working, not its accuracy."),
]

pair = Table([[image("fig4_confusion.png", 3.1 * inch), image("fig5_threshold.png", 2.92 * inch)],
              [Paragraph("Fig. 4: Held-out confusion matrix at the 60% threshold (one seed).", CAPTION),
               Paragraph("Fig. 5: Effect of the threshold (mean of five seeds). The dashed line marks the 60% used.", CAPTION)]],
             colWidths=[WIDTH / 2, WIDTH / 2])
pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, 0), "BOTTOM"), ("VALIGN", (0, 1), (-1, 1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
story += [
    KeepTogether(pair),
    p("<b>Results:</b> At the 60% threshold, 92.6% (± 1.0) of held-out signs were named correctly. Recall in "
      "Fig. 4 was 100% for Hello, Thank You and Yes, 98.8% for I Love You and 62.4% for Peace. Peace is partly "
      "understated: with Person B held out, only 33 Peace images remained for training. On the outside check, "
      "the letter V, which has the same handshape as Peace, was named Peace in 80% of 40 images. Fig. 5 shows "
      "that raising the threshold rejects few unfamiliar shapes (13.3% were still named as signs at 60%) while "
      "rejecting more real signs."),
    table([
        ["Measure", "Result"],
        ["Processing time per frame", "14 ms median, 26 ms at 95th percentile"],
        ["CPU and peak memory", "About 50% of one core; about 220 MB"],
        ["Classifier size and prediction time", "23 KB; about 11 microseconds"],
        ["Landmark extraction; evaluation and training", "70 s; 30 s"],
    ], [3.6 * inch, 2.9 * inch], "Table 2: Runtime on an M2 MacBook Air (8 GB RAM); per-frame figures from test video."),
    p("<b>Limitations:</b>"),
    bullets([
        "The system recognises static hand poses, not full sign language. Real Hello, Thank You and Yes use "
        "movement, and signing also uses the face, body and grammar.",
        "Testing covers about three people, unevenly (Fig. 6): Person A has no I Love You or Peace images. The "
        "model and threshold were also chosen on these held-out results, so expect lower accuracy on new hands.",
        "Testing so far used video files. A live webcam run is still pending.",
    ]),

    Spacer(1, 6),
    figure("fig6_people.png", "Fig. 6: Images used per sign and person, grouped approximately by clothing colour."),

    head("Learning Outcomes:"),
    bullets([
        "Learned hand landmark extraction using MediaPipe.",
        "Designed position- and scale-invariant features from keypoints.",
        "Trained and compared lightweight classifiers for real-time use.",
        "Learned why leave-one-person-out testing is needed to avoid data leakage.",
        "Applied confidence calibration and open-set rejection through a “no sign” class.",
        "Built a real-time computer vision application using OpenCV.",
    ]),

    head("References:"),
    numbered([
        "Sign_Language_Dataset, Mendeley Data, CC BY 4.0. doi.org/10.17632/pwyrdzwzwf.1",
        "SignAlphaSet, Mendeley Data, CC BY 4.0. doi.org/10.17632/8fmvr9m98w.3",
        "Google MediaPipe Hand Landmarker. ai.google.dev/edge/mediapipe",
    ]),
]

SimpleDocTemplate(OUT, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
                  title="Real-Time Hand Sign Detection", author="").build(story)
print("wrote", OUT)
