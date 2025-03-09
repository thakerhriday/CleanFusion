from utils.logger import Logger

if __name__ == "__main__":
    logger = Logger()
    logger.debug("🔎 Debug Log Test - This should appear in the log file")
    logger.info("✅ Info Log Test - This should appear in the log file")
    logger.warning("⚠️ Warning Log Test - This should appear in the log file")
    logger.error("❌ Error Log Test - This should appear in the log file")
    logger.critical("🔥 Critical Log Test - This should appear in the log file")
