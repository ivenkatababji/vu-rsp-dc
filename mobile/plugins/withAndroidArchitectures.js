const { withGradleProperties, createRunOncePlugin } = require('@expo/config-plugins');

const DEFAULT_ARCHITECTURES = 'arm64-v8a';

function normalizeArchitectures(value) {
  if (Array.isArray(value)) {
    return value.map(String).join(',');
  }
  if (typeof value === 'string' && value.trim()) {
    return value.trim();
  }
  return DEFAULT_ARCHITECTURES;
}

function withAndroidArchitectures(config, options = {}) {
  const architectures = normalizeArchitectures(options.architectures);

  return withGradleProperties(config, (gradleConfig) => {
    gradleConfig.modResults = gradleConfig.modResults.filter(
      (item) => !(item.type === 'property' && item.key === 'reactNativeArchitectures'),
    );
    gradleConfig.modResults.push({
      type: 'property',
      key: 'reactNativeArchitectures',
      value: architectures,
    });
    return gradleConfig;
  });
}

module.exports = createRunOncePlugin(withAndroidArchitectures, 'with-android-architectures', '1.0.0');
