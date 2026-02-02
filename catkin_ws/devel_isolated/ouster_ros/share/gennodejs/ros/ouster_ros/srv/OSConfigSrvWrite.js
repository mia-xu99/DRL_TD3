// Auto-generated. Do not edit!

// (in-package ouster_ros.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class OSConfigSrvWriteRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.metadata = null;
    }
    else {
      if (initObj.hasOwnProperty('metadata')) {
        this.metadata = initObj.metadata
      }
      else {
        this.metadata = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type OSConfigSrvWriteRequest
    // Serialize message field [metadata]
    bufferOffset = _serializer.string(obj.metadata, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type OSConfigSrvWriteRequest
    let len;
    let data = new OSConfigSrvWriteRequest(null);
    // Deserialize message field [metadata]
    data.metadata = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.metadata);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'ouster_ros/OSConfigSrvWriteRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd37888e5a47bef783c189dec5351027e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string metadata
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new OSConfigSrvWriteRequest(null);
    if (msg.metadata !== undefined) {
      resolved.metadata = msg.metadata;
    }
    else {
      resolved.metadata = ''
    }

    return resolved;
    }
};

class OSConfigSrvWriteResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.res = null;
    }
    else {
      if (initObj.hasOwnProperty('res')) {
        this.res = initObj.res
      }
      else {
        this.res = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type OSConfigSrvWriteResponse
    // Serialize message field [res]
    bufferOffset = _serializer.bool(obj.res, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type OSConfigSrvWriteResponse
    let len;
    let data = new OSConfigSrvWriteResponse(null);
    // Deserialize message field [res]
    data.res = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'ouster_ros/OSConfigSrvWriteResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e27848a10f8e7e4030443887dfea101b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool res
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new OSConfigSrvWriteResponse(null);
    if (msg.res !== undefined) {
      resolved.res = msg.res;
    }
    else {
      resolved.res = false
    }

    return resolved;
    }
};

module.exports = {
  Request: OSConfigSrvWriteRequest,
  Response: OSConfigSrvWriteResponse,
  md5sum() { return 'a9c18328c22f699f1b91d9e4167cda4a'; },
  datatype() { return 'ouster_ros/OSConfigSrvWrite'; }
};
