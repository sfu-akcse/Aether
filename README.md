# Aether

## project tool

We will be using [log.c](https://github.com/rxi/log.c) for logging purposes.

```c
log_trace(const char *fmt, ...);
log_debug(const char *fmt, ...);
log_info(const char *fmt, ...);
log_warn(const char *fmt, ...);
log_error(const char *fmt, ...);
log_fatal(const char *fmt, ...);
```

Each functions takes printf format string
```c
log_trace("Hello %s", "world")
```

Example code:
```c
#include "log.h"
#include <stdio.h>

int main() {
  int size = 5;
  int numbers[size];
  int sum = 0;
  log_set_level(LOG_TRACE);

  for (int i = 0; i < size; ++i) {
    log_trace("Loop iteration i = %d", i);
    numbers[i] = i;
    sum += numbers[i];
    log_info("Current sum is : %d", sum);
  }
  log_debug("Final sum is: %d", sum);

  printf("The sum is: %d", sum);
  return 0;
}
```