import os
from ohlcformer import logging
from unittest import TestCase
from logging import Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL


class TestLogging(TestCase):
    @staticmethod
    def set_os_environment(key, value):
        os.environ[key] = value

    @staticmethod
    def del_os_environment(key):
        try:
            del os.environ[key]
        except KeyError:
            return

    def test_set_logging_level(self):
        default_level = logging.get_default_logging_level()
        self.del_os_environment("OHLCFORMER_VERBOSITY")

        logging.set_logging_level("")
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), default_level)

        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), default_level)

        logging.set_logging_level("incorrect")
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), default_level)

        logging.set_logging_level('debug')
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), DEBUG)

        logging.set_logging_level('info')
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), INFO)

        logging.set_logging_level('warning')
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), WARNING)

        logging.set_logging_level('error')
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), ERROR)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "error")
        logging.set_logging_level('critical')
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), CRITICAL)

    def test_set_logging_level_with_env(self):
        default_level = logging.get_default_logging_level()

        self.del_os_environment("OHLCFORMER_VERBOSITY")
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), default_level)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), default_level)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "incorrect")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), default_level)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "debug")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), DEBUG)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "info")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), INFO)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "warning")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), WARNING)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "error")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), ERROR)

        self.set_os_environment("OHLCFORMER_VERBOSITY", "critical")
        logging.set_logging_level()
        logger = logging.get_logger()
        self.assertEqual(logger.getEffectiveLevel(), CRITICAL)

    def test_set_logging_formatting(self):
        default_formatter = logging.get_default_logging_formatter()
        self.del_os_environment("OHLCFORMER_LOG_FORMATTING")

        fmt = None
        logging.set_logging_formatting(fmt)
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, default_formatter._fmt)

        fmt = ""
        logging.set_logging_formatting(fmt)
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, "%(message)s")

        fmt = "%(levelname)s"
        logging.set_logging_formatting(fmt)
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)

        self.set_os_environment("OHLCFORMER_LOG_FORMATTING", "%(message)s")
        logging.set_logging_formatting(fmt)
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)

        self.assertRaises(TypeError, logging.set_logging_formatting, 0)
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)

    def test_set_logging_formatting_with_env(self):
        default_formatter = logging.get_default_logging_formatter()

        self.del_os_environment("OHLCFORMER_LOG_FORMATTING")
        logging.set_logging_formatting()
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, default_formatter._fmt)

        self.set_os_environment("OHLCFORMER_LOG_FORMATTING", "")
        logging.set_logging_formatting()
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, "%(message)s")

        fmt = "%(levelname)s"
        self.set_os_environment("OHLCFORMER_LOG_FORMATTING", fmt)
        logging.set_logging_formatting()
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)

        fmt = "%(message)s"
        self.set_os_environment("OHLCFORMER_LOG_FORMATTING", fmt)
        logging.set_logging_formatting(fmt)
        logger = logging.get_logger()
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)

    def test_get_logger(self):
        logger = logging.get_logger()
        self.assertEqual(logger.name, logging._get_lib_name())

        logger = logging.get_logger("test")
        self.assertEqual(logger.name, "test")

        self.assertRaises(TypeError, logging.get_logger, 0)

    def test_get_default_logging_level(self):
        default_level = logging.get_default_logging_level()
        self.assertEqual(type(default_level), int)

    def test_get_default_logging_formatter(self):
        default_formatter = logging.get_default_logging_formatter()
        self.assertEqual(type(default_formatter), Formatter)
