CXX        = g++
SIGN_ID    = Developer ID Application: Edward Lewis (Z9B4288ZRX)
ENTITLEMENTS = entitlements.plist
OCV_LIBS   = -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui
CXXFLAGS   = -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp -fobjc-arc \
             -I$(shell brew --prefix libomp)/include \
             -I$(shell brew --prefix opencv)/include/opencv4
LDFLAGS    = -L$(shell brew --prefix libomp)/lib -lomp \
             -L$(shell brew --prefix opencv)/lib $(OCV_LIBS) \
             -framework Vision -framework Foundation -framework CoreVideo

APP_NAME   = TimeMirror
APP_BUNDLE = $(APP_NAME).app
APP_MACOS  = $(APP_BUNDLE)/Contents/MacOS

all: time_mirror

time_mirror: time_mirror.mm
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
	@echo "Deduplicating LC_RPATH in dylibs..."
	@for f in dist/libs/*.dylib; do \
	  while install_name_tool -delete_rpath @executable_path/libs/ "$$f" 2>/dev/null; do :; done; \
	  install_name_tool -add_rpath @executable_path/libs/ "$$f" 2>/dev/null || true; \
	done
	zip -r9 time_mirror_mac.zip dist/
	@echo "Built: time_mirror_mac.zip ($$(du -sh time_mirror_mac.zip | cut -f1))"

app: dist
	rm -rf $(APP_BUNDLE)
	mkdir -p $(APP_MACOS)
	cp -r dist/libs $(APP_MACOS)/libs
	cp dist/time_mirror $(APP_MACOS)/time_mirror
	mkdir -p $(APP_BUNDLE)/Contents/Resources
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
	find $(APP_MACOS)/libs -name "*.dylib" -exec codesign --force --sign "$(SIGN_ID)" --timestamp --options runtime {} \;
	codesign --force --sign "$(SIGN_ID)" --timestamp --options runtime --entitlements $(ENTITLEMENTS) $(APP_MACOS)/time_mirror
	codesign --force --sign "$(SIGN_ID)" --timestamp --options runtime --entitlements $(ENTITLEMENTS) $(APP_BUNDLE)
	codesign --verify --deep --strict $(APP_BUNDLE) && echo "Signature OK"
	@echo "Built: $(APP_BUNDLE)"

app-zip: app
	zip -r9 $(APP_NAME).zip $(APP_BUNDLE)/
	@echo "Zipped: $(APP_NAME).zip ($$(du -sh $(APP_NAME).zip | cut -f1))"

NOTARY_PROFILE = timemirror

notarize-setup:
	xcrun notarytool store-credentials "$(NOTARY_PROFILE)" \
	  --apple-id "edlewis@gmail.com" \
	  --team-id Z9B4288ZRX

notarize: app-zip
	xcrun notarytool submit $(APP_NAME).zip \
	  --keychain-profile "$(NOTARY_PROFILE)" \
	  --wait
	xcrun stapler staple $(APP_BUNDLE)
	zip -r9 $(APP_NAME)_notarized.zip $(APP_BUNDLE)/
	@echo "Ready to distribute: $(APP_NAME)_notarized.zip ($$(du -sh $(APP_NAME)_notarized.zip | cut -f1))"

clean:
	rm -f time_mirror
	rm -rf dist time_mirror_mac.zip $(APP_BUNDLE) $(APP_NAME).zip $(APP_NAME)_notarized.zip

.PHONY: all run dist app app-zip clean
