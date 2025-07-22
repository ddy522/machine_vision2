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
                        DrawBoxes(json);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("서버 오류: " + ex.Message);
                }
            }
        }

        private void DrawBoxes(string json)
        {
            Dispatcher.Invoke(() =>
            {
                OverlayCanvas.Children.Clear();

                try
                {
                    var jobj = JObject.Parse(json);
                    var boxes = jobj["result"];

                    if (boxes == null) return;

                    foreach (var box in boxes)
                    {
                        double x1 = (double)box["x1"];
                        double y1 = (double)box["y1"];
                        double x2 = (double)box["x2"];
                        double y2 = (double)box["y2"];

                        double w = x2 - x1;
                        double h = y2 - y1;

                        var rect = new Rectangle
                        {
                            Width = w,
                            Height = h,
                            Stroke = Brushes.Red,
                            StrokeThickness = 2
                        };

                        Canvas.SetLeft(rect, x1);
                        Canvas.SetTop(rect, y1);

                        OverlayCanvas.Children.Add(rect);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("JSON 파싱 오류: " + ex.Message);
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
