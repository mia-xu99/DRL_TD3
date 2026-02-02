; Auto-generated. Do not edit!


(cl:in-package ouster_ros-srv)


;//! \htmlinclude OSConfigSrvWrite-request.msg.html

(cl:defclass <OSConfigSrvWrite-request> (roslisp-msg-protocol:ros-message)
  ((metadata
    :reader metadata
    :initarg :metadata
    :type cl:string
    :initform ""))
)

(cl:defclass OSConfigSrvWrite-request (<OSConfigSrvWrite-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <OSConfigSrvWrite-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'OSConfigSrvWrite-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ouster_ros-srv:<OSConfigSrvWrite-request> is deprecated: use ouster_ros-srv:OSConfigSrvWrite-request instead.")))

(cl:ensure-generic-function 'metadata-val :lambda-list '(m))
(cl:defmethod metadata-val ((m <OSConfigSrvWrite-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ouster_ros-srv:metadata-val is deprecated.  Use ouster_ros-srv:metadata instead.")
  (metadata m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <OSConfigSrvWrite-request>) ostream)
  "Serializes a message object of type '<OSConfigSrvWrite-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'metadata))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'metadata))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <OSConfigSrvWrite-request>) istream)
  "Deserializes a message object of type '<OSConfigSrvWrite-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'metadata) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'metadata) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<OSConfigSrvWrite-request>)))
  "Returns string type for a service object of type '<OSConfigSrvWrite-request>"
  "ouster_ros/OSConfigSrvWriteRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OSConfigSrvWrite-request)))
  "Returns string type for a service object of type 'OSConfigSrvWrite-request"
  "ouster_ros/OSConfigSrvWriteRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<OSConfigSrvWrite-request>)))
  "Returns md5sum for a message object of type '<OSConfigSrvWrite-request>"
  "a9c18328c22f699f1b91d9e4167cda4a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'OSConfigSrvWrite-request)))
  "Returns md5sum for a message object of type 'OSConfigSrvWrite-request"
  "a9c18328c22f699f1b91d9e4167cda4a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<OSConfigSrvWrite-request>)))
  "Returns full string definition for message of type '<OSConfigSrvWrite-request>"
  (cl:format cl:nil "string metadata~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'OSConfigSrvWrite-request)))
  "Returns full string definition for message of type 'OSConfigSrvWrite-request"
  (cl:format cl:nil "string metadata~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <OSConfigSrvWrite-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'metadata))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <OSConfigSrvWrite-request>))
  "Converts a ROS message object to a list"
  (cl:list 'OSConfigSrvWrite-request
    (cl:cons ':metadata (metadata msg))
))
;//! \htmlinclude OSConfigSrvWrite-response.msg.html

(cl:defclass <OSConfigSrvWrite-response> (roslisp-msg-protocol:ros-message)
  ((res
    :reader res
    :initarg :res
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass OSConfigSrvWrite-response (<OSConfigSrvWrite-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <OSConfigSrvWrite-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'OSConfigSrvWrite-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ouster_ros-srv:<OSConfigSrvWrite-response> is deprecated: use ouster_ros-srv:OSConfigSrvWrite-response instead.")))

(cl:ensure-generic-function 'res-val :lambda-list '(m))
(cl:defmethod res-val ((m <OSConfigSrvWrite-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ouster_ros-srv:res-val is deprecated.  Use ouster_ros-srv:res instead.")
  (res m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <OSConfigSrvWrite-response>) ostream)
  "Serializes a message object of type '<OSConfigSrvWrite-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'res) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <OSConfigSrvWrite-response>) istream)
  "Deserializes a message object of type '<OSConfigSrvWrite-response>"
    (cl:setf (cl:slot-value msg 'res) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<OSConfigSrvWrite-response>)))
  "Returns string type for a service object of type '<OSConfigSrvWrite-response>"
  "ouster_ros/OSConfigSrvWriteResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OSConfigSrvWrite-response)))
  "Returns string type for a service object of type 'OSConfigSrvWrite-response"
  "ouster_ros/OSConfigSrvWriteResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<OSConfigSrvWrite-response>)))
  "Returns md5sum for a message object of type '<OSConfigSrvWrite-response>"
  "a9c18328c22f699f1b91d9e4167cda4a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'OSConfigSrvWrite-response)))
  "Returns md5sum for a message object of type 'OSConfigSrvWrite-response"
  "a9c18328c22f699f1b91d9e4167cda4a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<OSConfigSrvWrite-response>)))
  "Returns full string definition for message of type '<OSConfigSrvWrite-response>"
  (cl:format cl:nil "bool res~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'OSConfigSrvWrite-response)))
  "Returns full string definition for message of type 'OSConfigSrvWrite-response"
  (cl:format cl:nil "bool res~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <OSConfigSrvWrite-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <OSConfigSrvWrite-response>))
  "Converts a ROS message object to a list"
  (cl:list 'OSConfigSrvWrite-response
    (cl:cons ':res (res msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'OSConfigSrvWrite)))
  'OSConfigSrvWrite-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'OSConfigSrvWrite)))
  'OSConfigSrvWrite-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OSConfigSrvWrite)))
  "Returns string type for a service object of type '<OSConfigSrvWrite>"
  "ouster_ros/OSConfigSrvWrite")