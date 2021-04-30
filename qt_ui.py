import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from skimage import io, feature, color, transform
from vp_dete_cali import main

class Ui_MainWindow(object):
    def __init__(self):       
        super(Ui_MainWindow, self).__init__()
        
    def setupUi(self, MainWindow):
        # Mainwindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 635)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # pushButton_openImage
        self.pushButton_openImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_openImage.setGeometry(QtCore.QRect(815, 10, 430, 50))
        self.pushButton_openImage.setObjectName("pushButton_openImage")
        # input height of camera
        self.heightLable = QtWidgets.QLabel(self.centralwidget)
        self.heightLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.heightLable.setGeometry(QtCore.QRect(950, 130, 295, 30))
        self.heightButton = QtWidgets.QPushButton(self.centralwidget)
        self.heightButton.setGeometry(QtCore.QRect(815, 70, 430, 50))
        self.heightButton.setObjectName("Input Height")
        # choose a ground point
        self.label_note = QtWidgets.QLabel(self.centralwidget)
        self.label_note.setGeometry(QtCore.QRect(820, 150, 430, 50))
        self.label_x = QtWidgets.QLabel(self.centralwidget)
        self.label_x.setGeometry(QtCore.QRect(820, 180, 430, 50))
        self.xLable = QtWidgets.QLabel(self.centralwidget)
        self.xLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.xLable.setGeometry(QtCore.QRect(1000, 190, 245, 29))
        self.label_y = QtWidgets.QLabel(self.centralwidget)
        self.label_y.setGeometry(QtCore.QRect(820, 210, 430, 50))
        self.yLable = QtWidgets.QLabel(self.centralwidget)
        self.yLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.yLable.setGeometry(QtCore.QRect(1000, 220, 245, 29))
        # pushButton_Calibration
        self.pushButton_Calibration = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Calibration.setGeometry(QtCore.QRect(815, 260, 430, 50))
        self.pushButton_Calibration.setObjectName("pushButton_Calibration")
        # label_image
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 800, 600))
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image.setObjectName("label_image")
        self.label_image.setScaledContents(True)
        # label_height
        self.label_height = QtWidgets.QLabel(self.centralwidget)
        self.label_height.setGeometry(QtCore.QRect(820, 120, 430, 50))
        # results:
        self.label_assume = QtWidgets.QLabel(self.centralwidget)
        self.label_assume.setGeometry(QtCore.QRect(820, 300, 430, 50))
        # f
        self.label_f = QtWidgets.QLabel(self.centralwidget)
        self.label_f.setGeometry(QtCore.QRect(820, 325, 430, 50))
        self.fLable = QtWidgets.QLabel(self.centralwidget)
        self.fLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.fLable.setGeometry(QtCore.QRect(820, 365, 425, 30))
        # r
        self.label_r = QtWidgets.QLabel(self.centralwidget)
        self.label_r.setGeometry(QtCore.QRect(820, 400, 430, 50))
        self.rLable = QtWidgets.QLabel(self.centralwidget)
        self.rLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.rLable.setGeometry(QtCore.QRect(820, 440, 425, 100))
        # t
        self.label_t = QtWidgets.QLabel(self.centralwidget)
        self.label_t.setGeometry(QtCore.QRect(820, 540, 430, 50))
        self.tLable = QtWidgets.QLabel(self.centralwidget)
        self.tLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.tLable.setGeometry(QtCore.QRect(820, 580, 425, 30))
        # menubar and statusbar
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # show name
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # clicked event
        self.pushButton_openImage.clicked.connect(self.openImage)
        self.pushButton_Calibration.clicked.connect(self.calibrationCamera)
        self.heightButton.clicked.connect(self.inputHeight)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", 'Vanishing Point Camera Calibration'))
        self.pushButton_openImage.setText(_translate("MainWindow", "Open Image"))
        self.pushButton_Calibration.setText(_translate("MainWindow", "Calibration"))
        self.heightButton.setText(_translate("MainWindow", "Input Height"))
        # self.heightLable.setText(_translate("MainWindow", "0"))
        self.label_height.setText(_translate("MainWindow", "Camera Height:"))
        self.label_note.setText(_translate("MainWindow", "(Please choose a point on the ground as origin point.)"))
        self.label_x.setText(_translate("MainWindow", "Point u in image (px):"))
        self.label_y.setText(_translate("MainWindow", "Point v in image (px):"))
        # self.xLable.setText(_translate("MainWindow", "0"))
        # self.yLable.setText(_translate("MainWindow", "0"))
        self.label_assume.setText(_translate("MainWindow", "(1.No distortion. 2.Principal point in the center.)"))
        self.label_f.setText(_translate("MainWindow", "Focal (px):"))
        self.label_r.setText(_translate("MainWindow", "Rotation:"))
        self.label_t.setText(_translate("MainWindow", "Translation (m):"))

    def openImage(self):  
        global imgName
        global rows_prop
        global cols_prop
        imgName = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Choose Image", "", "*.jpg;;*.png;;All Files(*)")[0]
        jpg = QtGui.QPixmap(imgName).scaled(self.label_image.width(), self.label_image.height())
        self.label_image.setPixmap(jpg)
        rows, cols = image_read(imgName)
        rows_prop = rows / self.label_image.height()
        cols_prop = cols / self.label_image.width()
        self.label_image.mousePressEvent = self.getPos

    def inputHeight(self):
        global h
        h, ok = QtWidgets.QInputDialog.getDouble(self.centralwidget, "Camera Height", "Please input camera height：", 0, 0, 20, 4)
        if ok :
            self.heightLable.setText(str(h))

    def getPos(self, event):
        global x
        global y
        global rows_prop
        global cols_prop
        x = event.pos().x() * cols_prop
        y = event.pos().y() * rows_prop
        _translate = QtCore.QCoreApplication.translate
        self.xLable.setText(_translate("MainWindow", str(x)))
        self.yLable.setText(_translate("MainWindow", str(y)))

    def calibrationCamera(self):
        global imgName
        global x
        global y
        global h
        px_x = x
        px_y = y
        cam_h = h
        F, M, V = main(imgName, px_x, px_y, cam_h)
        _translate = QtCore.QCoreApplication.translate
        self.fLable.setText(_translate("MainWindow", str(F[0])))
        self.rLable.setText(_translate("MainWindow", str(M[0])))
        self.tLable.setText(_translate("MainWindow", str(V[0])))

def image_read(imgpath):
    image = io.imread(imgName)
    rows = image.shape[0] 
    cols = image.shape[1]
    return rows, cols

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    obj = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(obj)
    obj.show()
    sys.exit(app.exec_())