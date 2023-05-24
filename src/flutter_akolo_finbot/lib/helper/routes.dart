// ignore_for_file: prefer_const_constructors

import 'package:flutter/material.dart';
//import 'package:googleauth/screens/welcome.screen.dart';
import 'customRoute.dart';
import 'package:flutter_akolo_finbot/ui/page/welcome.dart';
import 'package:flutter_akolo_finbot/ui/page/common/splash.dart';
import 'package:flutter_akolo_finbot/ui/page/auth/signin.dart';
import 'package:flutter_akolo_finbot/ui/page/auth/signup.dart';

class Routes {
  static dynamic route() {
    return {
      'SplashPage': (BuildContext context) => const SplashPage(),
      //'WelcomePage': (BuildContext context) => const WelcomePage(),
    };
  }

  static void sendNavigationEventToFirebase(String? path) {
    if (path != null && path.isNotEmpty) {
      // analytics.setCurrentScreen(screenName: path);
    }
  }

  static Route? onGenerateRoute(RouteSettings settings) {
    print("settings:" + settings.toString());
    final List<String> pathElements = settings.name!.split('/');
    print("pathElements:" + pathElements.toString());
    if (pathElements[0] != '' || pathElements.length == 1) {
      return null;
    }
    print("pathElements[1]:" + pathElements[1]);
    switch (pathElements[1]) {
      case "WelcomePage":
        return CustomRoute<bool>(
            builder: (BuildContext context) => const WelcomePage());
      case "SignIn":
        return CustomRoute<bool>(builder: (BuildContext context) => SignIn());
      case "SignUp":
        return CustomRoute<bool>(builder: (BuildContext context) => Signup());
      default:
        return onUnknownRoute(const RouteSettings(name: '/Feature'));
    }
  }

  static Route onUnknownRoute(RouteSettings settings) {
    return MaterialPageRoute(
      builder: (context) => Scaffold(
        appBar: AppBar(
          //title: customTitleText(
          //  settings.name!.split('/')[1],
          //),
          centerTitle: true,
        ),
        body: Center(
          child: Text('${settings.name!.split('/')[1]} Comming soon..'),
        ),
      ),
    );
  }
}
