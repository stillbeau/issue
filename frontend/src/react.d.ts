declare module 'react' {
  export const useState: any;
  export const useMemo: any;
  export const useEffect: any;
  const React: any;
  export default React;
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}
