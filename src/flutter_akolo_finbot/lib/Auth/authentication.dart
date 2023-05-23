import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:flutter_akolo_finbot/screens/home.screen.dart';
import 'package:flutter_akolo_finbot/screens/welcome.screen.dart';

FirebaseAuth auth = FirebaseAuth.instance;

const kIsWeb = false;

Future<void> twitterSignIn(context) async {
  TwitterAuthProvider twitterProvider = TwitterAuthProvider();

  if (kIsWeb) {
    await FirebaseAuth.instance.signInWithPopup(twitterProvider);
  } else {
    await FirebaseAuth.instance.signInWithProvider(twitterProvider);
    try {
      // Navigator.of(context).pushReplacement(WelcomeScreen())
      Navigator.push(
          context, MaterialPageRoute(builder: (context) => const HomeScreen()));
      print("successfully login!");
    } catch (e) {
      print("failed login!");
    }
  }
}

Future<void> googleSignIn(context) async {
  final gooleSignIn = GoogleSignIn();
  final googleSignInAccount = await gooleSignIn.signIn();
  if (googleSignInAccount != null) {
    final googleAuth = await googleSignInAccount.authentication;
    if (googleAuth.accessToken != null && googleAuth.idToken != null) {
      try {
        await auth.signInWithCredential(GoogleAuthProvider.credential(
            idToken: googleAuth.idToken, accessToken: googleAuth.accessToken));
        // Navigator.of(context).pushReplacement(WelcomeScreen())
        Navigator.push(context,
            MaterialPageRoute(builder: (context) => const HomeScreen()));
        print("successfully login!");
      } catch (e) {
        print("failed login!");
      }
    }
  }
}
