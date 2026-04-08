#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "logging/log.h"

/* File handle for the log file, kept open for the lifetime of the process. */
static FILE *g_log_file = NULL;

/* Create the logs/ directory if it does not already exist. */
static int ensure_log_directory(void) {
    if (mkdir("logs", 0755) == 0 || errno == EEXIST) {
        return 0;
    }

    fprintf(stderr, "Failed to create logs directory: %s\n", strerror(errno));
    return -1;
}

/* Flush and close the log file. Registered with atexit() so it runs
 * automatically when the process exits, ensuring no log records are lost. */
static void close_log_file(void) {
    if (g_log_file != NULL) {
        fflush(g_log_file);
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

/* Initialize the logging system.
 * Opens logs/aether-system.log in append mode and registers it as a
 * file callback with the rxi/log library (log.h). Also registers
 * close_log_file() via atexit() to flush and close the file on exit.
 * Returns 0 on success, -1 on failure. */
static int setup_logging(void) {
    if (ensure_log_directory() != 0) {
        return -1;
    }

    /* Open in append mode so logs from multiple runs are preserved. */
    g_log_file = fopen("logs/aether-system.log", "a");
    if (g_log_file == NULL) {
        fprintf(stderr, "Failed to open logs/aether-system.log: %s\n", strerror(errno));
        return -1;
    }

    if (log_add_fp(g_log_file, LOG_TRACE) != 0) {
        fprintf(stderr, "Failed to register file logger callback.\n");
        close_log_file();
        return -1;
    }

    /* Ensure the log file is flushed and closed on normal program exit. */
    atexit(close_log_file);
    log_set_level(LOG_TRACE);
    log_info("C logger initialized. Writing to logs/aether-system.log");
    return 0;
}

int print_hello() {
    log_info("Hello, World!");
    return 0;
}

int main() {
    if (setup_logging() != 0) {
        return 1;
    }

    log_info("Starting C entrypoint.");
    print_hello();
    log_info("C entrypoint shutdown complete.");
    return 0;
}
