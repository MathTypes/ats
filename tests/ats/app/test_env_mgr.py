from hydra import initialize, compose

from ats.app.env_mgr import EnvMgr


def test_env_mgr_init():
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(
            config_name="test_dev",
            overrides=[
                "job.train_start_date=2009-06-01",
                "job.eval_start_date=2009-06-04",
                "job.eval_end_date=2009-06-10",
                "job.test_start_date=2009-06-04",
                "job.eval_end_date=2009-06-10",
                "dataset.snapshot=''",
                "dataset.write_snapshot=False",
            ],
        )
        env_mgr = EnvMgr(cfg)
        # Sun May 31 2009 15:00:00
        assert env_mgr.train_start_timestamp == 1243807200.0
        # Wed Jun 03 2009 15:00:00
        assert env_mgr.eval_start_timestamp == 1244066400.0
        assert env_mgr.test_start_timestamp == 1244066400.0
        # Mon Aug 03 2009 14:00:00
        assert env_mgr.test_end_timestamp == 1249333200.0
