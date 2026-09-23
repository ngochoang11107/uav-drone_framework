# 1 node duoc colcon build trong ws, trong install cua ws co folder cung ten package chua folder libs chua code chay va folder share chua data 

#!/usr/bin/env python3

import rclpy            
#khởi tạo hệ thống ROS 2, quản lý vòng lặp thực thi (spin), và tương tác với mạng lưới ROS.

from rclpy.node import Node     
#Đây là một hàm tiện ích của hệ thống build ament trong ROS 2. Nó giúp bạn lấy đường dẫn tuyệt đối đến thư mục share của một package cụ thể.


import json     
# Dùng để phân tích (parse) và tạo ra dữ liệu chuẩn JSON. Trong robotics, nó thường được dùng để đọc các file cấu hình thông số của camera, hoặc lưu trữ/gửi dữ liệu tọa độ định dạng JSON.

import math     

import os       
# Cung cấp các hàm tương tác với hệ điều hành, ví dụ như kiểm tra xem một file có tồn tại không, nối các đoạn đường dẫn file lại với nhau

import cv2     
 # OpenCv, xử lý hình ảnh và video theo thời gian thực (lọc nhiễu, chuyển đổi không gian màu, phát hiện cạnh...).

import cv2.aruco as aruco       
#Một module con của OpenCV chuyên dùng để nhận diện mã ArUco (ArUco markers).

import numpy as np      
# tính toán mảng và ma trận đa chiều.

from ament_index_python.packages import get_package_share_directory
# lấy đường dẫn tuyệt đối đến thư mục share của một package cụ thể. Điều này rất quan trọng để code có thể tự động tìm thấy các file tài nguyên (file cấu hình, file yaml, mô hình 3D) 
# mà không cần phải ghi cứng (hardcode) đường dẫn trên máy tính của bạn.



#các hằng số cấu hình mặc định (default parameters), nơi lấy dữ liệu camera và loại marker cần tìm.


package_share = get_package_share_directory("aruco_detection")
#Tìm đường dẫn tuyệt đối trên máy tính trỏ đến thư mục share của package ROS 2 có tên là "aruco_detection".
# Thay vì bạn phải tự gõ một đường dẫn cứng nhắc (kiểu /home/user/ros2_ws/src/...), dòng này giúp ROS 2 tự động tìm đúng thư mục làm việc, 
# đảm bảo code chạy được trên bất kỳ máy tính nào mà không bị lỗi File Not Found.

DEFAULT_CALIB = os.path.join(package_share,"camera_calibration", "data", "elp210", "calib_data_mono.json")
#Sử dụng thư viện os để nối các thư mục lại với nhau, tạo ra đường dẫn hoàn chỉnh trỏ tới file calib_data_mono.json.
#File JSON này chứa các thông số nội tại của camera (intrinsic matrix) và hệ số méo ống kính (distortion coefficients) của loại camera tên là elp210

DEFAULT_DICT = "DICT_4x4_50" # da khai bao la 1 chuoi van ban
#Gán loại từ điển ArUco mặc định mà chương trình sẽ tìm kiếm là DICT_4x4_50.
#Marker 4x4, co 50 id ( 0 -> 49)


DEFAULT_MARKER_SIZE = 26.7
#Định nghĩa kích thước cạnh của hình vuông ArUco trong thế giới thực.
#Khi camera nhìn thấy một hình vuông, nó chỉ biết hình vuông đó dài bao nhiêu pixel trên màn hình. Nó cần biết kích thước thực tế (26.7 mm) 
# để thuật toán nội suy ra được marker đang nằm cách camera bao xa và xoay một góc bao nhiêu độ trong không gian 3D.


class Pose_estimation(Node):

    def __init__(self):
        super().__init__('pose_estimation_node')

        self.declare_parameter("marker_size", DEFAULT_MARKER_SIZE)
        self.declare_parameter("cam", 0)
        self.declare_parameter("id", -1)
        self.declare_parameter("calib", DEFAULT_CALIB)
        self.declare_parameter("side", "left")
    # self.declare_parameter("tên_tham_số_trong_ros", giá_trị_mặc_định)
    # Dòng này đăng ký với mạng lưới ROS 2 rằng Node của bạn sở hữu một tham số có tên là "marker_size".
    # Nếu khi khởi chạy Node, người dùng không truyền giá trị nào vào cho "marker_size", ROS 2 sẽ tự động sử dụng giá trị mặc định là DEFAULT_MARKER_SIZE (ví dụ: 26.7 mà bạn đã thiết lập ở trên).
    
    #ros2 run aruco_detection my_aruco_node --ros-args -p marker_size:=150.0 : thay vi phai vo code de chinh thong so thi chay dong lenh nay
      
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().value
        self.cam = self.get_parameter("cam").get_parameter_value().value
        self.id = self.get_parameter("id").get_parameter_value().value
        self.calib = self.get_parameter("calib").get_parameter_value().value
        self.side = self.get_parameter("side").get_parameter_value().value
    # self.get_parameter("marker_size"): Tìm và trả về một đối tượng Parameter của ROS 2. Đối tượng này chứa nhiều thông tin meta (kiểu dữ liệu, tên, giá trị...).
    # .get_parameter_value(): Lấy đối tượng ParameterValue từ tham số trên. ROS 2 dùng lớp này để bao bọc (wrap) các kiểu dữ liệu khác nhau (integer, double, boolean, string, mảng...).
    # .value: trích xuất dữ liệu thực sự bên trong thành một kiểu dữ liệu cơ bản của Python. Vì DEFAULT_MARKER_SIZE là số thực (26.7), nên .value ở đây sẽ trả về một số kiểu float.
        
        
        self.detect = self.make_detector(getattr(aruco, DEFAULT_DICT))
        # khởi tạo bộ nhận diện mã ArUco (ArUco Detector) dựa trên loại từ điển mặc định mà bạn đã cấu hình trước đó.
        # getattr (Get Attribute) là một hàm tích hợp sẵn của Python giúp bạn lấy một thuộc tính của object (hoặc module) thông qua một chuỗi văn bản.
        # getattr(đối_tượng, "tên_thuộc_tính")
        # Vì có tiền tố self, đối tượng nhận diện này được lưu giữ lại trong Node. Trong vòng lặp xử lý ảnh (mỗi khi camera có khung hình mới gửi về), 
        # bạn chỉ cần gọi self.detect.detectMarkers(image) để tìm mã ArUco thay vì phải khởi tạo lại từ đầu, giúp tiết kiệm tài nguyên và chạy thời gian thực (real-time) tốt hơn.
        # Người viết code đã dùng kỹ thuật Hàm bọc (Wrapper Function).
        # Bản thân OpenCV dùng hàm cv2.aruco.detectMarkers(...) để nhận diện. Tuy nhiên, hàm của OpenCV đòi hỏi phải truyền vào khá nhiều tham số lằng nhằng (như dictionary, parameters...).
        #Để code gọn gàng hơn, tác giả đã viết một hàm self.make_detector để "gói" tất cả các cài đặt đó lại, và trả về một hàm con gọn nhẹ hơn gán vào biến self.detect. Bây giờ self.detect hoạt động như một hàm có thể được gọi (callable).
        
        
        self.camera_matrix, self.dist_coeffs = self.load_calibration(self.calib, self.side)

        # "Mã ArUco này đang cách camera bao nhiêu mét? Đang nghiêng góc bao nhiêu độ?"
        # Để giải được bài toán đó, hàm của OpenCV bắt buộc phải nhận vào 4 yếu tố: Tọa độ của mã ArUco trên ảnh (pixel), Kích thước thực tế của mã ArUco (như bạn đã cài đặt 26.7 mm);
        #self.camera_matrix: Để nội suy khoảng cách, self.dist_coeffs: Để trừ hao độ cong của ống kính.
        
        self.cap = cv2.VidepCapture(self.cam)
        # truy cap vao cammera

        self.timer = self.create_timer(0.03, self.timer_callback)
        # Giá trị 0.03 giây tương đương với 30 mili-giây. Điều này có nghĩa là mỗi giây, hành động này sẽ được lặp lại khoảng 33 lần (33 Hz). 
        # Tần số này đặc biệt hoàn hảo để khớp với tốc độ khung hình tiêu chuẩn của camera (30 FPS - Frames Per Second).
        #Dòng code này có ý nghĩa là: "Hãy đặt báo thức, cứ đúng 0.03 giây lại đánh thức chương trình dậy để chạy hàm timer_callback một lần."



    def timer_callback(self):

        # Nhưng trong ROS 2, bạn KHÔNG ĐƯỢC PHÉP dùng while True.
        #Lý do là ROS 2 hoạt động dựa trên cơ chế hướng sự kiện (Event-driven). Ở cuối chương trình chính của ROS 2 luôn có một lệnh tên là rclpy.spin(node).
        # Lệnh spin() này liên tục lắng nghe mạng lưới để xem có ai gửi tin nhắn đến không, có ai gọi dịch vụ không.
        # Nếu bạn bỏ một vòng lặp while True vào trong Node, chương trình sẽ bị kẹt (blocking) vĩnh viễn ở vòng lặp đó. Nó sẽ mải mê đọc camera mà quên mất việc giao tiếp với các Node khác.
        # Hàm create_timer giải quyết triệt để vấn đề này: Nó chia nhỏ công việc ra. Cứ 0.03s, nó mượn luồng chính một chốc lát để xử lý ảnh,
        #  sau đó nhả luồng ra để ROS 2 làm các việc khác (như publish tọa độ của ArUco cho robot).
        # Neu ko co create timer, chuong trinh van se chay nhung ko lam gi ca
        # Chương trình sẽ không bị lỗi (không crash). Nó vẫn khởi tạo các thông số, vẫn kết nối với camera (cv2.VideoCapture), và vẫn báo cáo cho mạng lưới ROS 2 biết là Node này đang tồn tại.

        # 1. BẮT LẤY KHUNG HÌNH TỪ CAMERA
        ok, frame = self.cap.read()
        # Lệnh .read() sẽ chộp lấy đúng 1 bức ảnh tại thời điểm đó (lưu vào biến frame) để bạn đưa vào hàm nhận diện mã ArUco. 
        # ret là một biến boolean (True/False) báo hiệu xem việc chụp ảnh có thành công hay không.

        # 2. XỬ LÝ LỖI (Bắt buộc phải có)
        if not ok:
            self.get_logger().warning(f"cannot read frame")
            return
            # lệnh return rất quan trọng. Nếu camera bị lỏng cáp hoặc mất kết nối một nhịp, ret sẽ là False và frame sẽ trống rỗng. 
            # Nếu bạn không có lệnh return này, code sẽ tiếp tục chạy xuống Bước 3, cố gắng xử lý một bức ảnh không tồn tại 
            # và toàn bộ chương trình ROS 2 sẽ bị sập (crash) ngay lập tức với lỗi NoneType object has no attribute 'shape'.


        # 3. TIỀN XỬ LÝ ẢNH (Tối ưu hóa tốc độ)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # tất cả các camera thông thường (webcam, camera điện thoại, camera USB) đều luôn trả về ảnh 3 kênh màu BGR, bất kể bạn đang quay anh den hay xam.
        # để chuyển từ ảnh màu gốc sang ảnh xám 1 kênh nhằm tăng tốc độ tính toán cho ArUco, bạn phải dùng hằng số cv2.COLOR_BGR2GRAY chứ không phải RGB2GRAY.
        # Việc chuyển đổi này sẽ trộn 3 lớp (B, G, R) lại thành 1 lớp duy nhất (tương đương với cường độ sáng), giúp máy tính giảm được 2/3 khối lượng dữ liệu phải xử lý.
        

        # 4. NHẬN DIỆN MÃ ARUCO
        corners, ids, _ = self.detect(gray)
        # Hàm detectMarkers trả về 3 biến: 
            # - corners: 4 góc của các mã tìm được
            # - ids: Danh sách ID tương ứng
            # - rejected: Các hình vuông giống ArUco nhưng không hợp lệ


        # 5. XỬ LÝ KẾT QUẢ NẾU TÌM THẤY MÃ

        if ids is not None:
        #tvecs (Translation Vectors - Vector dịch tiến) : khoảng cách và vị trí (Tọa độ X, Y, Z).
            #X: Mã đang nằm lệch sang trái hay sang phải so với tâm camera.
            # Y: Mã đang nằm lệch lên trên hay xuống dưới so với tâm camera.
            #   Z: Khoảng cách trực tiếp từ camera đến bề mặt của mã (đây thường là con số robot quan tâm nhất để biết vật cản đang cách bao xa).
        #rvecs (Rotation Vectors - Vector xoay): Đây là góc nghiêng của mã trong không gian 3D. 
            # Dựa vào đây, robot có thể biết mặt phẳng chứa mã đang nằm ngửa lên trần nhà, úp xuống đất, hay quay thẳng vào ống kính (được biểu diễn dưới dạng các góc Roll, Pitch, Yaw).
            
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(

            #Hàm aruco.estimatePoseSingleMarkers(...) : tính toán xem mã ArUco đang nằm ở tọa độ nào trong không gian 3D và đang bị nghiêng góc bao nhiêu so với ống kính camera.
            # Nó sử dụng 4 "nguyên liệu" đầu vào mà chúng ta đã chuẩn bị từ các bước trước:
            
                corners,            #1. corners: Tọa độ 4 góc của hình vuông ArUco trên bức ảnh (tính bằng pixel, thu được từ hàm detectMarkers).
                self.args.markers_size,     #2. self.args.markers_size: Kích thước vật lý của mã (ví dụ: 26.7 mm). Nó đóng vai trò làm "thước đo" để thuật toán quy đổi từ pixel sang milimet.
                self.camera_matrix,
                self.dist_coeffs        #3. self.camera_matrix & 4. self.dist_coeffs: Các thông số đã đọc từ file JSON giúp thuật toán nắn thẳng quang sai và nội suy khoảng cách.
        )
        aruco.drawDetectedMarkers(frame, corners, ids)
        # Hàm này sẽ vẽ 4 đường viền (thường là màu xanh lá) nối 4 góc của marker và in số ID (màu đỏ) tương ứng lên bức ảnh frame gốc.

        y= 30


        # Đoạn code này hoạt động như một bộ lọc (filter). Chức năng chính của nó là duyệt qua tất cả các mã ArUco mà camera nhìn thấy,
        # nhưng chỉ chọn ra đúng cái mã có ID mà bạn đang muốn tìm để xử lý, và phớt lờ tất cả các mã còn lại.
        for i, marker_id in enumerate(ids.flatten()):
            # ids: Là kết quả trả về từ thuật toán nhận diện trước đó (chứa danh sách các ID tìm thấy trong ảnh). 
            # Tuy nhiên, OpenCV trả về biến ids dưới dạng ma trận cột 2 chiều (ví dụ: [[5], [17], [42]]).
            #.flatten(): Là một hàm của thư viện Numpy. Nó có tác dụng "đập phẳng" ma trận 2 chiều thành một mảng 1 chiều bình thường (ví dụ biến thành: [5, 17, 42]). Điều này giúp vòng lặp Python duyệt qua dễ dàng hơn.
            # enumerate(...): Hàm này sinh ra một cặp giá trị cho mỗi lần lặp:
            #i (Index): Số thứ tự của mã trong mảng (0, 1, 2...). Số i này rất quan trọng để lát nữa bạn gọi đúng vị trí góc corners[i] hoặc khoảng cách tvecs[i] tương ứng.
            #marker_id: Giá trị ID thực sự của mã đó (ví dụ: 5, 17, 42).
            if self.id is not None and marker_id != self.id:
                # self.id is not None: Kiểm tra xem người dùng có thực sự cài đặt ID mục tiêu hay không. 
                # Nếu giá trị này là None, nghĩa là người dùng không quan tâm cụ thể ID nào, camera thấy mã nào thì xử lý mã đó.
                #marker_id != self.id: Kiểm tra xem ID của mã mà camera vừa đọc được có KHÁC với ID mục tiêu hay không.
                continue    # Hãy bỏ qua tất cả các dòng code phía dưới, và quay lại đầu vòng lặp để chuyển sang phần tử tiếp theo."



            rvec = rvecs[i]
            tvec = tvecs[i]

            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 0.5)
            # Trong OpenCV, màu sắc luôn tuân theo quy tắc RGB tương ứng với XYZ.
            # trực quan hóa (visualize) hệ trục tọa độ 3D của mã ArUco vừa tính toán được và vẽ đè nó lên bức ảnh 2D ban đầu để bạn có thể nhìn thấy trên màn hình.
           # Tại sao lại nhân với 0.5? Việc lấy một nửa kích thước của cạnh marker sẽ giúp 3 đường trục được vẽ ra với độ dài vừa vặn, bắt đầu từ tâm (0,0) và vươn dài ra vừa đúng chạm đến mép của hình vuông ArUco, trông rất thẩm mỹ và dễ quan sát.
            
            x, yy, z = tvec.ravel()
            # .ravel(): Là một hàm của thư viện NumPy, dùng để "đập phẳng" một ma trận đa chiều thành một mảng 1 chiều. 
            # Biến tvec ban đầu có dạng ma trận cột [[x], [y], [z]], sau khi dùng .ravel() sẽ trở thành [x, y, z].
            # x: Khoảng cách lệch sang trái/phải.
            # yy: Khoảng cách lệch lên/xuống (việc đặt tên là yy thay vì y có thể do người viết code muốn tránh trùng tên với một biến y nào đó đã có trước trong chương trình).
            # z: Chiều sâu từ camera đâm thẳng ra marker.

            distance = np.linalg.norm(tvec)
            # tính khoảng cách đường chim bay (Euclidean distance) từ tâm camera đến tâm của mã ArUco.
            # np.linalg.norm(): Hàm của NumPy dùng để tính độ lớn (magnitude) của một vector. Về mặt toán học, nó thực hiện công thức căn bậc hai tổng bình phương 3 trục
            
            
            rmat, _ = cv2.Rodrigues(rvec)

            # Biến rvec (Rotation Vector) sinh ra từ OpenCV là một vector 3x1 cực kỳ tối ưu cho máy tính, nhưng con người không thể đọc trực tiếp các con số trong đó để hiểu là nó nghiêng bao nhiêu độ.
            # Hàm cv2.Rodrigues() sử dụng công thức toán học Rodrigues để biến đổi vector 3x1 này thành một Ma trận xoay 3x3 (Rotation Matrix - rmat).

            roll, pitch, yaw = self.rotation_to_euler(rmat)
            # một hàm tự viết (custom function) bên trong Node của bạn. Nó nhận đầu vào là ma trận xoay 3x3 ở trên và tính toán ra 3 góc Euler quen thuộc


            lines = [
                f"id={marker_id} dist={distance:.1f} cm",
                f"x={x:.1f} y={yy:.1f} z={z:.1f} cm",
                f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg",
            ]

            for line in lines:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 2,)
                y+=25
            y+=10
        else:
            cv2.putText(frame,"Marker not found",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2,)

        cv2.imshow("aruco_pose_estimation", frame) # hien thi anh tren cua so co ten la aruco_pose_estimation

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
            return
        
    def load_calibration(self, path, side):
        with open(path) as f:
            data = json.load(f)
        # with open(path) as f:: Mở file cấu hình (nằm ở đường dẫn path). Việc dùng từ khóa with là một thói quen lập trình (best practice) rất tốt trong Python, 
        # nó đảm bảo file sẽ tự động được đóng lại một cách an toàn sau khi đọc xong, tránh rò rỉ bộ nhớ.
        # data = json.load(f): Dùng thư viện json để dịch toàn bộ nội dung file văn bản đó thành một Dictionary (từ điển) của Python và lưu vào biến data.
        
        camera_matrix = np.array(data[side]["matrix"], dtype = np.float64)
        # data[side]["matrix"]: Truy cập vào từ điển data để lấy mảng chứa ma trận. Biến side ở đây thường là chuỗi "mono" (nếu dùng 1 camera), hoặc "left", "right" (nếu dùng camera kép stereo).
        # np.array(..., dtype = np.float64): OpenCV không hiểu kiểu dữ liệu List (danh sách) cơ bản của Python. Do đó, dòng này chuyển đổi danh sách đó thành một ma trận NumPy. 
        # Tham số dtype = np.float64 ép kiểu dữ liệu thành số thực có độ chính xác cao (64-bit float) vì các hệ số quang học yêu cầu độ chính xác tới nhiều chữ số thập phân.
        
        dist_coeffs = np.array(data[side]["distorsion"], dtype = np.float64).reshape(-1,1)
        # truy cập vào file JSON để lấy mảng hệ số méo
        # .reshape(-1, 1): Hệ số méo lấy từ JSON thường là một mảng nằm ngang (ví dụ: [k1, k2, p1, p2, k3]). 
        # Tuy nhiên, các hàm tính toán 3D của OpenCV lại ưu tiên nhận dữ liệu đầu vào là một ma trận cột (1 cột và nhiều hàng).
        # Lệnh .reshape(-1, 1) nói với máy tính rằng: "Tôi muốn có ma trận 1 cột, còn số hàng () thì hãy tự động tính toán (số -1) dựa trên số lượng phần tử đang có". Kết quả là mảng ngang sẽ được dựng đứng lên thành mảng dọc.
        return camera_matrix, dist_coeffs

    def make_detector(self, dict_id):
        if hasattr(aruco, "ArucoDetector"):
            dictionary = aruco.getPredefineDictionary(dict_id)
            detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
            return detector.detectMarkers
        
        dictionary = aruco.Dictionary_get(dict_id)
        parameters = aruco.DetectorParameters_create()
        return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=parameters)
        # Bất chấp máy tính của bạn cài OpenCV phiên bản nào, hàm này đều "xào nấu" lại các lệnh khác biệt và trả ra một kết quả duy nhất. 
        # Nhờ vậy, ở phần vòng lặp chính của chương trình, bạn chỉ cần gọi một câu lệnh duy nhất là corners, ids, _ = self.detect(gray) mà không bao giờ sợ bị lỗi hệ thống.
    
    def rotation_to_euler(self, rmat): # Chuyển đổi Ma trận xoay 3x3 thành các góc Euler (Roll, Pitch, Yaw).
        # # 1. Tính toán hệ số Singularity (để kiểm tra Gimbal Lock
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)  
        
        if sy > 1e-6:
            roll = math.atan2(rmat[2, 1], rmat[2, 2]) # Tính bằng cách so sánh R_21 và R_22.
            pitch = math.atan2(-rmat[2, 0], sy)  # Tính bằng cách so sánh -R_20 và biến sy (cạnh huyền của R_00 và R_10).
            yaw = math.atan2(rmat[1, 0], rmat[0, 0]) # Tính bằng cách so sánh R_10 và R_00

        else: #Góc Pitch bị xoay vuông góc 90 độ hoặc -90 độ, # Nếu sy quá nhỏ gần bằng 0, nghĩa là bị Gimbal Lock
            roll = math.atan2(-rmat[1, 2], rmat[1, 1])
            pitch = math.atan2(-rmat[2, 0], sy)
            yaw = 0.0
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def destroy_node(self): 
        #Trước khi xóa Node này, hãy chạy thêm một số lệnh dọn dẹp riêng của tôi đã."
        # self.cap thường là một đối tượng cv2.VideoCapture() được khai báo trước đó để kết nối với camera hoặc đọc file video.
        # isOpened(): Kiểm tra xem luồng camera/video có đang mở và hoạt động hay không.
        # release(): Giải phóng tài nguyên camera. Lệnh này cực kỳ quan trọng. Nếu bạn tắt chương trình mà không giải phóng, 
        # phần cứng camera có thể bị "treo" (bị chiếm dụng ngầm), khiến các ứng dụng khác (hoặc chính code của bạn khi chạy lại) không thể mở camera lên được nữa.
        if self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindowns()

        # Lệnh này có nhiệm vụ đóng tất cả các cửa sổ giao diện GUI mà OpenCV đã tạo ra (thường là các cửa sổ hiển thị hình ảnh từ lệnh cv2.imshow()). 
        # Việc này giúp tránh hiện tượng các cửa sổ bị kẹt (zombie windows) trên màn hình sau khi tắt terminal.
        super().destroy_node()
        # super() đại diện cho lớp cha (ở đây là class Node gốc của ROS 2).
        #Sau khi bạn đã tự dọn dẹp xong phần camera và cửa sổ của OpenCV, lệnh này sẽ gọi hàm destroy_node() nguyên bản của ROS 2 để hệ thống 
        # làm nốt các việc còn lại như: ngắt kết nối các Publishers, Subscribers, Timers, và dọn dẹp bộ nhớ theo đúng chuẩn quy trình.
def main(args=None):
    rclpy.init(args=args)
    node = Pose_estimation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()