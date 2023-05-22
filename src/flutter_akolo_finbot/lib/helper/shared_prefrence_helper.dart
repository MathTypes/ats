//import 'package:flutter_akolo_finbot/model/link_media_info.dart';
import 'dart:convert';
//import 'package:flutter_akolo_finbot/model/user.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SharedPreferenceHelper {
  SharedPreferenceHelper._internal();
  static final SharedPreferenceHelper _singleton =
      SharedPreferenceHelper._internal();

  factory SharedPreferenceHelper() {
    return _singleton;
  }

  Future<String?> getUserName() async {
    return (await SharedPreferences.getInstance())
        .getString(UserPreferenceKey.UserName.toString());
  }

  Future clearPreferenceValues() async {
    (await SharedPreferences.getInstance()).clear();
  }
}

enum UserPreferenceKey { AccessToken, UserProfile, UserName, IsFirstTimeApp }
