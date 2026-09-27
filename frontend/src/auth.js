import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL?.trim();
const publishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY?.trim();

export const isSupabaseConfigured = Boolean(supabaseUrl && publishableKey);

// 管理员令牌只保存在当前标签页会话中，不落入长期 localStorage。
export const supabase = isSupabaseConfigured
  ? createClient(supabaseUrl, publishableKey, {
      auth: {
        storage: window.sessionStorage,
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: false,
      },
    })
  : null;

/** 使用现有 Supabase 账号登录；游客不能注册，也不能自行授予管理员权限。 */
export async function signInAdmin(email, password) {
  if (!supabase) throw new Error("管理员登录尚未配置，请联系网站管理员。");

  const { data, error } = await supabase.auth.signInWithPassword({ email, password });
  if (error || !data.session?.access_token) {
    throw new Error("邮箱或密码不正确，或该账号暂不可登录。");
  }
  return data.session.access_token;
}

export async function signOutAdmin() {
  if (!supabase) return;
  const { error } = await supabase.auth.signOut({ scope: "local" });
  if (error) throw new Error("退出登录失败，请检查网络后重试。");
}

export async function getAccessToken() {
  if (!supabase) return "";
  const { data, error } = await supabase.auth.getSession();
  if (error) return "";
  return data.session?.access_token || "";
}
