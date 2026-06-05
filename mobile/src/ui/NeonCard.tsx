import { PropsWithChildren } from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';
import { theme } from '../theme/tokens';

type NeonCardProps = PropsWithChildren<{
  style?: ViewStyle;
  tone?: 'default' | 'flat' | 'accent';
}>;

export function NeonCard({ children, style, tone = 'default' }: NeonCardProps) {
  return (
    <View style={[styles.card, tone === 'flat' && styles.flat, tone === 'accent' && styles.accent, style]}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: theme.radii.lg,
    backgroundColor: theme.colors.surfaceElevated,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    ...theme.shadows.soft,
  },
  flat: {
    backgroundColor: theme.colors.surfaceContainerLow,
    shadowOpacity: 0,
    elevation: 0,
  },
  accent: {
    backgroundColor: theme.colors.surfaceContainer,
    borderColor: theme.colors.primarySoft,
  },
});
