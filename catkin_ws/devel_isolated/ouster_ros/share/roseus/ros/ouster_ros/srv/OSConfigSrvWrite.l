;; Auto-generated. Do not edit!


(when (boundp 'ouster_ros::OSConfigSrvWrite)
  (if (not (find-package "OUSTER_ROS"))
    (make-package "OUSTER_ROS"))
  (shadow 'OSConfigSrvWrite (find-package "OUSTER_ROS")))
(unless (find-package "OUSTER_ROS::OSCONFIGSRVWRITE")
  (make-package "OUSTER_ROS::OSCONFIGSRVWRITE"))
(unless (find-package "OUSTER_ROS::OSCONFIGSRVWRITEREQUEST")
  (make-package "OUSTER_ROS::OSCONFIGSRVWRITEREQUEST"))
(unless (find-package "OUSTER_ROS::OSCONFIGSRVWRITERESPONSE")
  (make-package "OUSTER_ROS::OSCONFIGSRVWRITERESPONSE"))

(in-package "ROS")





(defclass ouster_ros::OSConfigSrvWriteRequest
  :super ros::object
  :slots (_metadata ))

(defmethod ouster_ros::OSConfigSrvWriteRequest
  (:init
   (&key
    ((:metadata __metadata) "")
    )
   (send-super :init)
   (setq _metadata (string __metadata))
   self)
  (:metadata
   (&optional __metadata)
   (if __metadata (setq _metadata __metadata)) _metadata)
  (:serialization-length
   ()
   (+
    ;; string _metadata
    4 (length _metadata)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _metadata
       (write-long (length _metadata) s) (princ _metadata s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _metadata
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _metadata (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass ouster_ros::OSConfigSrvWriteResponse
  :super ros::object
  :slots (_res ))

(defmethod ouster_ros::OSConfigSrvWriteResponse
  (:init
   (&key
    ((:res __res) nil)
    )
   (send-super :init)
   (setq _res __res)
   self)
  (:res
   (&optional (__res :null))
   (if (not (eq __res :null)) (setq _res __res)) _res)
  (:serialization-length
   ()
   (+
    ;; bool _res
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _res
       (if _res (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _res
     (setq _res (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass ouster_ros::OSConfigSrvWrite
  :super ros::object
  :slots ())

(setf (get ouster_ros::OSConfigSrvWrite :md5sum-) "a9c18328c22f699f1b91d9e4167cda4a")
(setf (get ouster_ros::OSConfigSrvWrite :datatype-) "ouster_ros/OSConfigSrvWrite")
(setf (get ouster_ros::OSConfigSrvWrite :request) ouster_ros::OSConfigSrvWriteRequest)
(setf (get ouster_ros::OSConfigSrvWrite :response) ouster_ros::OSConfigSrvWriteResponse)

(defmethod ouster_ros::OSConfigSrvWriteRequest
  (:response () (instance ouster_ros::OSConfigSrvWriteResponse :init)))

(setf (get ouster_ros::OSConfigSrvWriteRequest :md5sum-) "a9c18328c22f699f1b91d9e4167cda4a")
(setf (get ouster_ros::OSConfigSrvWriteRequest :datatype-) "ouster_ros/OSConfigSrvWriteRequest")
(setf (get ouster_ros::OSConfigSrvWriteRequest :definition-)
      "string metadata
---
bool res

")

(setf (get ouster_ros::OSConfigSrvWriteResponse :md5sum-) "a9c18328c22f699f1b91d9e4167cda4a")
(setf (get ouster_ros::OSConfigSrvWriteResponse :datatype-) "ouster_ros/OSConfigSrvWriteResponse")
(setf (get ouster_ros::OSConfigSrvWriteResponse :definition-)
      "string metadata
---
bool res

")



(provide :ouster_ros/OSConfigSrvWrite "a9c18328c22f699f1b91d9e4167cda4a")


