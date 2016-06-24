package br.com.munif.wsdlspike.facerecjavacv;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.indexer.DoubleBufferIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Buffer;
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
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_ROUGH_SEARCH;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_FIND_BIGGEST_OBJECT;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameConverter;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.javacpp.opencv_core.cvCopy;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

public class Programa extends JFrame implements ActionListener {

    final private static int WEBCAM_DEVICE_INDEX = 0;

    private JLabel labelImagem;
    private JLabel labelImagem2;
    private JLabel labelImagem3;
    private JTextField tfEmail;
    private VideoCapture capture;
    private int absoluteFaceSize;
    private Mat destino;
    private CvHaarClassifierCascade classifier;
    private CvMemStorage storage = null;
    private CvSeq faces = null;
    private FaceRecognizer faceRecognizer;
    private JCheckBox reconhece;
    private Map<Integer, String> imagens;
    private BufferedImage image2;

    private String lastImage = "NONE";

    public Programa() {
        super("Face");

        setLayout(new FlowLayout());
        JButton btCaptura = new JButton("Caputra");
        reconhece = new JCheckBox("Reconhece", false);
        btCaptura.addActionListener(this);
        add(btCaptura);
        add(reconhece);
        labelImagem = new JLabel("Antes");
        add(labelImagem);

        labelImagem2 = new JLabel("Depois");
        add(labelImagem2);
        labelImagem3 = new JLabel("Quem");
        add(labelImagem3);
        setPreferredSize(new Dimension(800, 600));
        tfEmail = new JTextField(50);
        add(tfEmail);
        pack();

        setDefaultCloseOperation(EXIT_ON_CLOSE);
        initClassifier();
        initRecognizer();
        initCapture();
        initServer();
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (image2 != null) {
            try {
                ImageIO.write(image2, "JPG", new File("/home/munif/Desktop/caras/" + tfEmail.getText() + "_" + System.currentTimeMillis() + ".jpg"));
            } catch (IOException ex) {
                Logger.getLogger(Programa.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

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
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                g2.drawRect(r.x(), r.y(), r.width(), r.height());

                cvSetImageROI(gray, r);
                IplImage cropped = cvCreateImage(cvGetSize(gray), gray.depth(), gray.nChannels());
                cvCopy(gray, cropped);

                IplImage resizeImage = IplImage.create(200, 256, cropped.depth(), cropped.nChannels());
                cvResize(cropped, resizeImage);

                Frame finalImage = grabberConverter.convert(resizeImage);

                image2 = paintConverter.getBufferedImage(finalImage);
                labelImagem2.setIcon(new ImageIcon(image2));

                OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
                Mat convert = converterToMat.convert(finalImage);
                if (reconhece.isSelected()) {
                    Mat m2 = convert.clone();
                    //System.out.println("----m2----" + m2.arrayWidth() + "x" + m2.arrayHeight() + "x" + m2.arrayDepth());
                    int ib[] = new int[1];
                    double db[] = new double[1];
                    faceRecognizer.predict(m2, ib, db);
                    //faceRecognizer.
                    setTitle("" + db[0]);
                    labelImagem3.setIcon(new ImageIcon(imagens.get(ib[0])));
                    lastImage = "" + imagens.get(ib[0]);
                }
            }
            faces = null;
        }
        labelImagem.setIcon(new ImageIcon(image));

    }

    public static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), 3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.data().put(data);
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
        imagens = new HashMap<>();
        File root = new File("/home/munif/Desktop/caras");
        //File root = new File("/Users/munif/Downloads/BioID-FaceDatabase-V1.2/");

        FilenameFilter filtro = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".jpeg");
            }
        };
        File[] imageFiles = root.listFiles(filtro);
        if (imageFiles.length == 0) {
            return;
        }
        MatVector caras = new MatVector(imageFiles.length);
        int counter = 0;
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            caras.put(counter, img);
            labelsBuf.put(counter, counter);
            imagens.put(counter, image.getAbsolutePath());
            counter++;
            System.out.printf("%05d ", counter);
            if (counter % 20 == 0) {
                System.out.println("");
            }
            if (counter > 1000) {
                break;
            }

        }
        //faceRecognizer = createFisherFaceRecognizer();
        //faceRecognizer = createEigenFaceRecognizer();
        faceRecognizer = createLBPHFaceRecognizer();
        //faceRecognizer = createFisherFaceRecognizer();

        faceRecognizer.train(caras, labels);
    }

    public void initServer() {
        try {
            HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
            server.createContext("/id", new JsonHandler());
            server.createContext("/image", new ImageHandler());
            server.setExecutor(null); // creates a default executor
            server.start();
        } catch (IOException ex) {
            Logger.getLogger(Programa.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    class JsonHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange he) throws IOException {
            String response = "{\"image\":\"" + lastImage + "\"}";
            he.sendResponseHeaders(200, response.getBytes().length);
            OutputStream os = he.getResponseBody();
            os.write(response.getBytes());
            os.close();
        }
    }

    class ImageHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange he) throws IOException {
            ByteArrayOutputStream baos=new ByteArrayOutputStream();
            ImageIO.write(image2, lastImage, baos);
            he.getResponseHeaders().set("Content-Type","image/jpeg");
            he.sendResponseHeaders(200, baos.size());
            OutputStream os = he.getResponseBody();
            os.write(baos.toByteArray());
            os.flush();
            os.close();
        }
    }

}
