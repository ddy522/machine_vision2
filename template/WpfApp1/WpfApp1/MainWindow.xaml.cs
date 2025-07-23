using Newtonsoft.Json.Linq; // JSON 파싱용
using OpenCvSharp;
using System.IO;
using System.Net.Http;
using System.Windows;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using OpenCvSharp.WpfExtensions;
using System.Windows.Controls;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Collections.Generic;
using System;

namespace WpfApp1
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture capture;      // OpenCV 웹캠
        private Mat frame;                 // 현재 프레임
        private bool isRunning = true;     // 루프 제어
        private readonly HttpClient _httpClient = new HttpClient(); // YOLO 서버 요청

        private DateTime _lastSent = DateTime.MinValue; // 요청 간격 제어

        // 이전 박스 저장
        private List<System.Windows.Rect> _previousBoxes = new List<System.Windows.Rect>();

        public MainWindow()
        {
            InitializeComponent();

            capture = new VideoCapture(0); // 0번 카메라
            frame = new Mat();

            if (!capture.IsOpened())
            {
                MessageBox.Show("웹캠을 찾을 수 없습니다.");
                return;
            }

            // 1) 프레임 루프 돌리기
            Task.Run(FrameLoop);

            // 2) WPF 데이터 그리드에 JSON 데이터 불러오기
            LoadGridDataAsync();
        }

        private async Task FrameLoop()
        {
            while (isRunning)
            {
                capture.Read(frame);
                if (!frame.Empty())
                {
                    // WPF에 표시
                    var image = frame.ToBitmapSource();
                    image.Freeze();
                    Dispatcher.Invoke(() =>
                    {
                        WebcamImage.Source = image;
                    });

                    // YOLO에 주기적으로 보내기 (0.5초 간격)
                    if ((DateTime.Now - _lastSent).TotalMilliseconds > 500)
                    {
                        _lastSent = DateTime.Now;
                        await SendFrameToServer(frame);
                    }
                }
            }
        }

        private async Task SendFrameToServer(Mat mat)
        {
            using (var ms = mat.ToMemoryStream(".jpg"))
            {
                var content = new MultipartFormDataContent();
                content.Add(new StreamContent(ms), "file", "frame.jpg");

                try
                {
                    var response = await _httpClient.PostAsync("http://127.0.0.1:8000/predict", content);
                    if (response.IsSuccessStatusCode)
                    {
                        var json = await response.Content.ReadAsStringAsync();
                        System.Diagnostics.Debug.WriteLine(json);
                        DrawBoxes(json);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("서버 오류: " + ex.Message);
                }
            }
        }

        private bool AreBoxesSimilar(List<System.Windows.Rect> boxes1, List<System.Windows.Rect> boxes2, double tolerance = 3.0)
        {
            if (boxes1.Count != boxes2.Count)
                return false;

            for (int i = 0; i < boxes1.Count; i++)
            {
                var b1 = boxes1[i];
                var b2 = boxes2[i];

                if (Math.Abs(b1.X - b2.X) > tolerance ||
                    Math.Abs(b1.Y - b2.Y) > tolerance ||
                    Math.Abs(b1.Width - b2.Width) > tolerance ||
                    Math.Abs(b1.Height - b2.Height) > tolerance)
                {
                    return false;
                }
            }
            return true;
        }

        private void DrawBoxes(string json)
        {
            Dispatcher.Invoke(() =>
            {
                try
                {
                    var jobj = JObject.Parse(json);
                    var results = jobj["result"];

                    double yoloWidth = (double)jobj["image_width"];
                    double yoloHeight = (double)jobj["image_height"];

                    double canvasWidth = OverlayCanvas.ActualWidth;
                    double canvasHeight = OverlayCanvas.ActualHeight;

                    double scaleX = canvasWidth / yoloWidth;
                    double scaleY = canvasHeight / yoloHeight;

                    if (results == null) return;

                    OverlayCanvas.Children.Clear();

                    foreach (var result in results)
                    {
                        var points = result["points"];
                        int pointCount = points.Count();

                        if (pointCount < 3)
                        {
                            // 점 3개 미만이면 그리지 않음
                            continue;
                        }

                        List<System.Drawing.Point> pts = new List<System.Drawing.Point>();
                        foreach (var p in points)
                        {
                            double rawX = (double)p[0];
                            double rawY = (double)p[1];

                            if (rawX < 0) rawX = 0; // 클램핑
                            if (rawY < 0) rawY = 0;

                            double x = rawX * scaleX;
                            double y = rawY * scaleY;

                            pts.Add(new System.Drawing.Point((int)x, (int)y));
                        }

                        // 중심 기준 각도 정렬 (꼭짓점 순서 보정)
                        double centerX = pts.Average(pt => pt.X);
                        double centerY = pts.Average(pt => pt.Y);

                        pts.Sort((a, b) =>
                        {
                            double angleA = System.Math.Atan2(a.Y - centerY, a.X - centerX);
                            double angleB = System.Math.Atan2(b.Y - centerY, b.X - centerX);
                            return angleA.CompareTo(angleB);
                        });

                        var polygon = new Polygon
                        {
                            Stroke = Brushes.Red,
                            StrokeThickness = 2,
                            Fill = Brushes.Transparent,
                            Points = new PointCollection((IEnumerable<System.Windows.Point>)pts)
                        };

                        OverlayCanvas.Children.Add(polygon);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("JSON 파싱 오류: " + ex.Message);
                }
            });
        }



        // ★ 추가: 데이터 그리드에 서버 JSON 불러오기 ★
        private async Task LoadGridDataAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("http://127.0.0.1:8000/get_data");
                response.EnsureSuccessStatusCode();

                var jsonString = await response.Content.ReadAsStringAsync();

                // 서버에서 받은 JSON을 Bom 리스트로 디시리얼라이즈
                var dataList = JsonConvert.DeserializeObject<List<Bom>>(jsonString);

                // DataGridTop에 바인딩 (필요하면 필터링 가능)
                Dispatcher.Invoke(() =>
                {
                    DataGridTop.ItemsSource = dataList;
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show("데이터 그리드 로드 실패: " + ex.Message);
            }
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            isRunning = false;
            capture?.Release();
            frame?.Dispose();
            base.OnClosing(e);
        }
    }

    // 편의 확장 메서드
    public static class MatExtensions
    {
        public static MemoryStream ToMemoryStream(this Mat mat, string ext = ".jpg")
        {
            Cv2.ImEncode(ext, mat, out var buf);
            return new MemoryStream(buf);
        }
    }
}
