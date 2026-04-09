CXX        = g++
OCV_LIBS   = -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui
CXXFLAGS   = -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
             -I$(shell brew --prefix libomp)/include \
             -I$(shell brew --prefix opencv)/include/opencv4
LDFLAGS    = -L$(shell brew --prefix libomp)/lib -lomp \
             -L$(shell brew --prefix opencv)/lib $(OCV_LIBS)

APP_NAME   = TimeMirror
APP_BUNDLE = $(APP_NAME).app
APP_MACOS  = $(APP_BUNDLE)/Contents/MacOS

all: time_mirror

time_mirror: time_mirror.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: time_mirror
	./time_mirror

dist: time_mirror
	rm -rf dist
	mkdir -p dist/libs
	cp time_mirror dist/
	dylibbundler -b -x dist/time_mirror -d dist/libs/ -p @executable_path/libs/
	cp /opt/homebrew/lib/libtbb.12.17.dylib dist/libs/
	install_name_tool -id @executable_path/libs/libtbb.12.17.dylib dist/libs/libtbb.12.17.dylib
	@echo "Stripping duplicate LC_RPATH from dylibs..."
	@for f in dist/libs/*.dylib; do \
	  install_name_tool -delete_rpath @executable_path/libs/ "$$f" 2>/dev/null || true; \
	done
	zip -r9 time_mirror_mac.zip dist/
	@echo "Built: time_mirror_mac.zip ($$(du -sh time_mirror_mac.zip | cut -f1))"

app: dist
	rm -rf $(APP_BUNDLE)
	mkdir -p $(APP_MACOS)
	cp -r dist/libs $(APP_MACOS)/libs
	cp dist/time_mirror $(APP_MACOS)/time_mirror
	find $(APP_MACOS)/libs -name "*.dylib" -exec codesign --force --sign - {} \;
	codesign --force --sign - $(APP_MACOS)/time_mirror
	codesign --force --sign - $(APP_BUNDLE)
	xattr -cr $(APP_BUNDLE)
	@printf '<?xml version="1.0" encoding="UTF-8"?>\n\
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n\
<plist version="1.0"><dict>\n\
  <key>CFBundleName</key><string>$(APP_NAME)</string>\n\
  <key>CFBundleExecutable</key><string>time_mirror</string>\n\
  <key>CFBundleIdentifier</key><string>com.edlewis.timemirror</string>\n\
  <key>CFBundleVersion</key><string>1.0</string>\n\
  <key>CFBundlePackageType</key><string>APPL</string>\n\
  <key>NSCameraUsageDescription</key><string>Required for webcam capture</string>\n\
  <key>NSHighResolutionCapable</key><true/>\n\
</dict></plist>\n' > $(APP_BUNDLE)/Contents/Info.plist
	@echo "Built: $(APP_BUNDLE)"

app-zip: app
	zip -r9 $(APP_NAME).zip $(APP_BUNDLE)/
	@echo "Zipped: $(APP_NAME).zip ($$(du -sh $(APP_NAME).zip | cut -f1))"

clean:
	rm -f time_mirror
	rm -rf dist time_mirror_mac.zip $(APP_BUNDLE) $(APP_NAME).zip

.PHONY: all run dist app app-zip clean
