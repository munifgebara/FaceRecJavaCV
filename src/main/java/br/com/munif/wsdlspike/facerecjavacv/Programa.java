package br.com.munif.wsdlspike.facerecjavacv;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.net.URL;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.bytedeco.javacpp.Loader;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.CvType;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.cvCopy;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_ROUGH_SEARCH;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_FIND_BIGGEST_OBJECT;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

public class Programa extends JFrame implements ActionListener {

    final private static int WEBCAM_DEVICE_INDEX = 0;

    private JLabel labelImagem;
    private JLabel labelImagem2;
    private VideoCapture capture;
    private int absoluteFaceSize;
    private Mat destino;
    private CvHaarClassifierCascade classifier;
    private CvMemStorage storage = null;
    private CvSeq faces = null;
    private FaceRecognizer faceRecognizer;

    public Programa() {
        super("Face");
        initClassifier();
        initRecognizer();
        setLayout(new FlowLayout());
        JButton btCaptura = new JButton("Caputra");
        btCaptura.addActionListener(this);
        add(btCaptura);
        labelImagem = new JLabel("Antes");

        add(labelImagem);
        labelImagem2 = new JLabel("Depois");
        add(labelImagem2);
        pack();

        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        initCapture();
    }

    private void initCapture() {
        capture = new VideoCapture(WEBCAM_DEVICE_INDEX);

        Runnable frameGrabber = new Runnable() {
            @Override
            public void run() {
                if (capture.isOpened()) {
                    Mat frame = new Mat();
                    capture.read(frame);
                    if (!frame.empty()) {
                        detect(frame);
                    }
                }
            }
        };

        ScheduledExecutorService newSingleThreadScheduledExecutor = Executors.newSingleThreadScheduledExecutor();
        ScheduledFuture<?> scheduleWithFixedDelay = newSingleThreadScheduledExecutor.scheduleWithFixedDelay(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

        System.out.println("------>" + scheduleWithFixedDelay.isCancelled());

    }

    @Override
    protected void finalize() throws Throwable {
        this.capture.release();
        super.finalize(); //To change body of generated methods, choose Tools | Templates.
    }
    OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
    Java2DFrameConverter paintConverter = new Java2DFrameConverter();

    private void detect(Mat mat) {

        Frame frame = grabberConverter.convert(mat);

        IplImage colorIPl = grabberConverter.convert(frame);

        BufferedImage image = paintConverter.getBufferedImage(frame);

        IplImage gray = IplImage.create(frame.imageWidth, frame.imageHeight, opencv_core.IPL_DEPTH_8U, 1);
        org.bytedeco.javacpp.opencv_imgproc.cvCvtColor(colorIPl, gray, CV_BGR2GRAY);
        faces = cvHaarDetectObjects(gray, classifier, storage, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH);

        if (faces != null) {
            Graphics2D g2 = image.createGraphics();
            g2.setColor(Color.RED);
            g2.setStroke(new BasicStroke(2));
            int total = faces.total();
            System.out.println("---->" + total);
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                g2.drawRect(r.x(), r.y(), r.width(), r.height());

                cvSetImageROI(colorIPl, r);
                IplImage cropped = cvCreateImage(cvGetSize(colorIPl), colorIPl.depth(), colorIPl.nChannels());

                cvCopy(colorIPl, cropped);
                IplImage resizeImage = IplImage.create(75, 100, cropped.depth(), cropped.nChannels());
                cvResize(cropped, resizeImage);

                Frame finalImage = grabberConverter.convert(resizeImage);

                BufferedImage image2 = paintConverter.getBufferedImage(finalImage);
                labelImagem2.setIcon(new ImageIcon(image2));
                byte[] pixels = ((DataBufferByte) image2.getRaster().getDataBuffer()).getData();
                Mat imgMat = new Mat();
                //imgMat.put(0, 0, pixels);

                int predictedLabel = faceRecognizer.predict(imgMat);
                System.out.println("Predicted label: " + predictedLabel);

            }
            faces = null;
        }
        labelImagem.setIcon(new ImageIcon(image));

    }

    

public static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), 3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        //mat.put(0, 0, data);
        return mat;
    }

    public void initClassifier() {
        try {
            URL url = new URL("https://raw.github.com/Itseez/opencv/2.4.0/data/haarcascades/haarcascade_frontalface_alt.xml");
            File file = Loader.extractResource(url, null, "classifier", ".xml");
            file.deleteOnExit();
            String classifierName = file.getAbsolutePath();
            System.out.println("---->" + classifierName);
            Loader.load(opencv_objdetect.class);
            classifier = new CvHaarClassifierCascade(cvLoad(classifierName));
            storage = CvMemStorage.create();

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public static void main(String[] args) {
        new Programa().setVisible(true);
    }

    private void initRecognizer() {
        File root = new File("/home/munif/caras");
        FilenameFilter filtro = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".png") || name.endsWith(".jpeg");
            }
        };
        File[] imageFiles = root.listFiles(filtro);
        MatVector caras = new MatVector(imageFiles.length);
        int counter = 0;

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            caras.put(counter, img);
            counter++;
        }
        faceRecognizer = createFisherFaceRecognizer();
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        faceRecognizer.train(caras, labels);
    }

}
