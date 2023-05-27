import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:flutter_akolo_finbot/helper/utility.dart';
import 'package:flutter_akolo_finbot/state/authState.dart';
import 'package:flutter_akolo_finbot/widgets/newWidget/customLoader.dart';
import 'package:flutter_akolo_finbot/widgets/newWidget/rippleButton.dart';
import 'package:flutter_akolo_finbot/widgets/newWidget/title_text.dart';

class GoogleLoginButton extends StatelessWidget {
  const GoogleLoginButton({
    Key? key,
    required this.loader,
    this.loginCallback,
  }) : super(key: key);
  final CustomLoader loader;
  final Function? loginCallback;
  void _googleLogin(context) {
    var state = Provider.of<AuthState>(context, listen: false);
    if (state.user != null) {
      Utility.logEvent("_googleLogin, origal_state:" + state.user.toString());
    } else {
      Utility.logEvent("_googleLogin, origal_state: null user");
    }
    Utility.logEvent("_googleLogin, context:" + context.toString());
    loader.showLoader(context);
    state.handleGoogleSignIn().then((status) {
      Utility.logEvent("_googleLogin, status:" + status.toString());
      if (state.user != null) {
        Utility.logEvent("_googleLogin, after_state:" + state.user.toString());
      } else {
        Utility.logEvent("_googleLogin, after_state: null user");
      }
      if (state.user != null) {
        loader.hideLoader();
        Navigator.pop(context);
        Utility.logEvent("calling callback");
        if (loginCallback != null) loginCallback!();
      } else {
        loader.hideLoader();
        Utility.logEvent('Unable to login');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return RippleButton(
      onPressed: () {
        _googleLogin(context);
      },
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: const <BoxShadow>[
            BoxShadow(
              color: Color(0xffeeeeee),
              blurRadius: 15,
              offset: Offset(5, 5),
            ),
          ],
        ),
        child: Wrap(
          children: <Widget>[
            Image.asset(
              'assets/images/google_logo.png',
              height: 20,
              width: 20,
            ),
            const SizedBox(width: 10),
            const TitleText(
              'Continue with Google',
              color: Colors.black54,
            ),
          ],
        ),
      ),
    );
  }
}
