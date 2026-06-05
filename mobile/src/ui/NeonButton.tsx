import { Pressable, StyleProp, StyleSheet, Text, View, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { theme } from '../theme/tokens';

type NeonButtonProps = {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'quiet';
  icon?: keyof typeof MaterialCommunityIcons.glyphMap;
  style?: StyleProp<ViewStyle>;
};

export function NeonButton({
  label,
  onPress,
  disabled = false,
  variant = 'primary',
  icon,
  style,
}: NeonButtonProps) {
  const isPrimary = variant === 'primary';
  const iconColor =
    variant === 'primary'
      ? theme.colors.onPrimary
      : variant === 'quiet'
        ? theme.colors.onSurfaceMuted
        : theme.colors.onSurface;
  const labelStyle =
    variant === 'primary'
      ? styles.primaryLabel
      : variant === 'quiet'
        ? styles.quietLabel
        : styles.secondaryLabel;

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled }}
      style={({ pressed }) => [
        styles.base,
        variant === 'secondary' && styles.secondary,
        variant === 'quiet' && styles.quiet,
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
        style,
      ]}
    >
      {isPrimary ? (
        <LinearGradient
          colors={[theme.colors.primary, theme.colors.primaryContainer]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.gradient}
        >
          {icon ? <MaterialCommunityIcons name={icon} size={18} color={iconColor} /> : null}
          <Text style={styles.primaryLabel}>{label}</Text>
        </LinearGradient>
      ) : (
        <View style={styles.inlineContent}>
          {icon ? <MaterialCommunityIcons name={icon} size={18} color={iconColor} /> : null}
          <Text style={labelStyle}>{label}</Text>
        </View>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    borderRadius: theme.radii.lg,
    overflow: 'hidden',
    minHeight: 48,
  },
  gradient: {
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: theme.spacing.sm,
    minHeight: 48,
  },
  primaryLabel: {
    color: theme.colors.onPrimary,
    fontFamily: 'Manrope_700Bold',
    fontSize: 15,
    letterSpacing: 0.1,
  },
  secondary: {
    backgroundColor: theme.colors.surfaceContainerHigh,
    borderWidth: 1,
    borderColor: theme.colors.outlineStrong,
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 48,
  },
  quiet: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  inlineContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.sm,
  },
  secondaryLabel: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 14,
    letterSpacing: 0.1,
  },
  quietLabel: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_700Bold',
    fontSize: 14,
    letterSpacing: 0.1,
  },
  disabled: {
    opacity: 0.46,
  },
  pressed: {
    transform: [{ scale: 0.98 }],
  },
});
